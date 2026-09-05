from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from prepare_vllm_frozen_drafter_overlay import (  # noqa: E402
    prepare_overlay,
    prerequisite_patches,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(root: Path) -> tuple[Path, Path, Path]:
    package = root / "site-packages" / "vllm"
    target = package / "runtime.py"
    target.parent.mkdir(parents=True)
    target.write_text("mode = 'stock'\n", encoding="utf-8")

    patch_path = root / "runtime.patch"
    patch_path.write_text(
        """diff --git a/vllm/runtime.py b/vllm/runtime.py
--- a/vllm/runtime.py
+++ b/vllm/runtime.py
@@ -1 +1 @@
-mode = 'stock'
+mode = 'refit-aware'
""",
        encoding="utf-8",
    )
    policy_module = root / "frozen_drafter_sleep_policy.py"
    policy_module.write_text("DRAFT_WEIGHT_TAG = 'draft'\n", encoding="utf-8")
    return package, patch_path, policy_module


def test_prepare_overlay_is_immutable_and_records_exact_inputs(tmp_path: Path) -> None:
    source, patch_path, policy_module = write_fixture(tmp_path)

    overlay_package = prepare_overlay(
        source_package=source,
        overlay_root=tmp_path / "overlay",
        runtime_patch_path=patch_path,
        expected_runtime_patch_sha256=sha256(patch_path),
        policy_module_path=policy_module,
        expected_policy_module_sha256=sha256(policy_module),
    )

    assert (source / "runtime.py").read_text(encoding="utf-8") == "mode = 'stock'\n"
    assert (overlay_package / "runtime.py").read_text(encoding="utf-8") == (
        "mode = 'refit-aware'\n"
    )
    installed_policy = (
        overlay_package / "device_allocator" / "nemo_rl_frozen_drafter_sleep.py"
    )
    assert installed_policy.read_bytes() == policy_module.read_bytes()

    receipt = json.loads(
        (tmp_path / "overlay" / "frozen-drafter-sleep-overlay.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["schema_version"] == 1
    assert receipt["runtime_patch_sha256"] == sha256(patch_path)
    assert receipt["policy_module_sha256"] == sha256(policy_module)
    assert receipt["status"] == "applied"


def test_prepare_overlay_rejects_unpinned_policy_module(tmp_path: Path) -> None:
    source, patch_path, policy_module = write_fixture(tmp_path)

    with pytest.raises(ValueError, match="policy module digest mismatch"):
        prepare_overlay(
            source_package=source,
            overlay_root=tmp_path / "overlay",
            runtime_patch_path=patch_path,
            expected_runtime_patch_sha256=sha256(patch_path),
            policy_module_path=policy_module,
            expected_policy_module_sha256="0" * 64,
        )

    assert not (tmp_path / "overlay").exists()


def test_dspark_overlay_resolves_both_pinned_runtime_prerequisites(
    tmp_path: Path,
) -> None:
    patches = prerequisite_patches(tmp_path, dspark_enabled=True)

    assert [path.name for path, _ in patches] == [
        "vllm-0.25.1-pr48167-runtime.patch",
        "vllm-0.25.1-pr48167-group-causality-followup.patch",
    ]
    assert [digest for _, digest in patches] == [
        "504730a52614fddeb8ea899ec37a0aa820dcbc3a57c704fc13f5834fcc07b317",
        "8e5ff0e385ee44cf71e1e07031e5cd19658b29eb7b90bc172a4754c599d1dd90",
    ]
    assert prerequisite_patches(tmp_path, dspark_enabled=False) == ()
