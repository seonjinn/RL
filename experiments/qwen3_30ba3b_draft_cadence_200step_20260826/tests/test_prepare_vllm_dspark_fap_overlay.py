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
STOCK_RUNNER = (
    "self.speculator.init_cudagraph_manager(mode)\nself.speculator.set_attn()\n"
)
PATCHED_RUNNER = (
    "self.speculator.set_attn()\nself.speculator.init_cudagraph_manager(mode)\n"
)
STOCK_CAUSALITY = "causal=self.dflash_causal\n"
PATCHED_CAUSALITY = "causal=self._group_causal\n"


def write_package(
    root: Path,
    guard: str,
    runner: str,
    causality: str = STOCK_CAUSALITY,
) -> Path:
    package = root / "site-packages" / "vllm"
    backend = package / "v1" / "attention" / "backends" / "flashinfer.py"
    backend.parent.mkdir(parents=True)
    backend.write_text(f"before\n{guard}after\n")
    runner_path = package / "v1" / "worker" / "gpu" / "model_runner.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text(runner)
    speculator = (
        package / "v1" / "worker" / "gpu" / "spec_decode" / "dflash" / "speculator.py"
    )
    speculator.parent.mkdir(parents=True)
    speculator.write_text(causality * 2)
    (package / "__init__.py").write_text('__version__ = "0.25.1"\n')
    return package


def write_runtime_patch(root: Path) -> Path:
    patch = root / "pr48167-runtime.patch"
    patch.write_text(
        """diff --git a/vllm/v1/attention/backends/flashinfer.py b/vllm/v1/attention/backends/flashinfer.py
--- a/vllm/v1/attention/backends/flashinfer.py
+++ b/vllm/v1/attention/backends/flashinfer.py
@@ -1,4 +1,5 @@
 before
-        if has_trtllm_support:
+        # trtllm-gen only supports causal attention.
+        if has_trtllm_support and not vllm_config.attention_config.use_non_causal:
             return AttentionCGSupport.UNIFORM_BATCH
 after
diff --git a/vllm/v1/worker/gpu/model_runner.py b/vllm/v1/worker/gpu/model_runner.py
--- a/vllm/v1/worker/gpu/model_runner.py
+++ b/vllm/v1/worker/gpu/model_runner.py
@@ -1,2 +1,2 @@
-self.speculator.init_cudagraph_manager(mode)
 self.speculator.set_attn()
+self.speculator.init_cudagraph_manager(mode)
"""
    )
    return patch


def patch_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_causality_patch(root: Path) -> Path:
    patch = root / "group-causality.patch"
    patch.write_text(
        """diff --git a/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py b/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
--- a/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
+++ b/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
@@ -1,2 +1,2 @@
-causal=self.dflash_causal
-causal=self.dflash_causal
+causal=self._group_causal
+causal=self._group_causal
"""
    )
    return patch


def test_parse_args_uses_job_local_overlay_environment_for_post_sync(
    tmp_path: Path,
) -> None:
    overlay_root = tmp_path / "overlay"

    args = parse_args([], {"Q30_VLLM_OVERLAY": str(overlay_root)})

    assert args.source_package is None
    assert args.overlay_root == overlay_root


def test_parse_args_fails_closed_without_cli_or_environment_overlay() -> None:
    with pytest.raises(SystemExit):
        parse_args([], {})


def test_prepare_overlay_applies_complete_runtime_patch_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = write_package(tmp_path / "source", STOCK_GUARD, STOCK_RUNNER)
    patch = write_runtime_patch(tmp_path)
    followup = write_causality_patch(tmp_path)
    source_files = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    overlay_package = prepare_overlay(
        source,
        tmp_path / "overlay",
        patch,
        expected_patch_sha256=patch_sha256(patch),
        followup_patch_path=followup,
        expected_followup_patch_sha256=patch_sha256(followup),
    )

    assert (
        overlay_package / "v1" / "attention" / "backends" / "flashinfer.py"
    ).read_text() == f"before\n{PATCHED_GUARD}after\n"
    assert (
        overlay_package / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text() == PATCHED_RUNNER
    assert {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == source_files
    receipt = json.loads(
        (tmp_path / "overlay" / "dspark-fap-vllm-48167-runtime.json").read_text()
    )
    assert receipt["schema_version"] == 3
    assert receipt["upstream_pr"] == "https://github.com/vllm-project/vllm/pull/48167"
    assert receipt["status"] == "applied"
    assert receipt["patch_sha256"] == patch_sha256(patch)
    assert set(receipt["patched_files"]) == {
        "vllm/v1/attention/backends/flashinfer.py",
        "vllm/v1/worker/gpu/model_runner.py",
    }
    assert receipt["followup_status"] == "applied"
    assert receipt["followup_patch_sha256"] == patch_sha256(followup)
    assert set(receipt["followup_patched_files"]) == {
        "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
    }
    assert (
        overlay_package
        / "v1"
        / "worker"
        / "gpu"
        / "spec_decode"
        / "dflash"
        / "speculator.py"
    ).read_text() == PATCHED_CAUSALITY * 2


def test_prepare_overlay_accepts_complete_already_patched_source(
    tmp_path: Path,
) -> None:
    source = write_package(
        tmp_path / "source",
        PATCHED_GUARD,
        PATCHED_RUNNER,
        PATCHED_CAUSALITY,
    )
    patch = write_runtime_patch(tmp_path)
    followup = write_causality_patch(tmp_path)

    prepare_overlay(
        source,
        tmp_path / "overlay",
        patch,
        expected_patch_sha256=patch_sha256(patch),
        followup_patch_path=followup,
        expected_followup_patch_sha256=patch_sha256(followup),
    )

    receipt = json.loads(
        (tmp_path / "overlay" / "dspark-fap-vllm-48167-runtime.json").read_text()
    )
    assert receipt["status"] == "already-patched"
    assert receipt["followup_status"] == "already-patched"


def test_prepare_overlay_rejects_source_drift(tmp_path: Path) -> None:
    source = write_package(
        tmp_path / "source",
        "        if has_trtllm_support and unknown_condition:\n"
        "            return AttentionCGSupport.UNIFORM_BATCH\n",
        STOCK_RUNNER,
    )
    patch = write_runtime_patch(tmp_path)

    with pytest.raises(ValueError, match="does not apply cleanly"):
        prepare_overlay(
            source,
            tmp_path / "overlay",
            patch,
            expected_patch_sha256=patch_sha256(patch),
        )

    assert not (tmp_path / "overlay").exists()


def test_prepare_overlay_rejects_unpinned_patch(tmp_path: Path) -> None:
    source = write_package(tmp_path / "source", STOCK_GUARD, STOCK_RUNNER)
    patch = write_runtime_patch(tmp_path)

    with pytest.raises(ValueError, match="runtime patch digest mismatch"):
        prepare_overlay(
            source,
            tmp_path / "overlay",
            patch,
            expected_patch_sha256="0" * 64,
        )

    assert not (tmp_path / "overlay").exists()
