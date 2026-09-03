from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PREPARER = EXPERIMENT_DIR / "prepare_mcore_checkpoint_overlay.py"
PRODUCTION_PATCH = EXPERIMENT_DIR / "patches/mcore-precision-aware-lazy-state-checkpoint.patch"

SOURCE_TEXT = """\
class DistributedOptimizer:
    def sharded_state_dict(self, sharding_type, is_loading):
        if not is_loading and sharding_type == 'fully_sharded_bucket_space':
            self.warn()

        state_dict = self.state_dict()
        return state_dict
"""

PATCH_TEXT = """\
diff --git a/megatron/core/optimizer/distrib_optimizer.py b/megatron/core/optimizer/distrib_optimizer.py
--- a/megatron/core/optimizer/distrib_optimizer.py
+++ b/megatron/core/optimizer/distrib_optimizer.py
@@ -6,2 +6,8 @@ class DistributedOptimizer:
-        state_dict = self.state_dict()
+        if (
+            not is_loading
+            and self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
+        ):
+            self.init_state_fn(self.optimizer, self.config)
+
+        state_dict = self.state_dict()
         return state_dict
"""

PATCHED_TEXT = """\
class DistributedOptimizer:
    def sharded_state_dict(self, sharding_type, is_loading):
        if not is_loading and sharding_type == 'fully_sharded_bucket_space':
            self.warn()

        if (
            not is_loading
            and self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        ):
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()
        return state_dict
"""

PRODUCTION_SOURCE_TEXT = """\
                ' Please switch to `full_sharded_model_space`.',
            )

        state_dict = self.state_dict()
        if sharding_type not in self.checkpoint_fully_reshardable_formats:
            # State dict differs between different model parallel groups
"""

PRODUCTION_PATCHED_TEXT = """\
                ' Please switch to `full_sharded_model_space`.',
            )

        if (
            not is_loading
            and self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        ):
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()
        if sharding_type not in self.checkpoint_fully_reshardable_formats:
            # State dict differs between different model parallel groups
"""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_preparer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_mcore_checkpoint_overlay", PREPARER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "Megatron-LM"
    target = source_root / "megatron/core/optimizer/distrib_optimizer.py"
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE_TEXT)
    (source_root / "megatron/__init__.py").write_text("")
    patch_path = tmp_path / "checkpoint.patch"
    patch_path.write_text(PATCH_TEXT)
    return source_root, tmp_path / "overlay", patch_path


def prepare_fixture(preparer: ModuleType, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_root, overlay_root, patch_path = write_fixture(tmp_path)
    receipt_path = preparer.prepare_overlay(
        source_root=source_root,
        overlay_root=overlay_root,
        patch_path=patch_path,
        expected_source_sha256=sha256_bytes(SOURCE_TEXT.encode()),
        expected_patch_sha256=sha256_bytes(PATCH_TEXT.encode()),
        expected_patched_sha256=sha256_bytes(PATCHED_TEXT.encode()),
    )
    return source_root, overlay_root, patch_path, receipt_path


def test_prepare_overlay_applies_patch_and_writes_bound_receipt(tmp_path: Path) -> None:
    preparer = load_preparer()
    source_root, overlay_root, patch_path = write_fixture(tmp_path)

    receipt_path = preparer.prepare_overlay(
        source_root=source_root,
        overlay_root=overlay_root,
        patch_path=patch_path,
        expected_source_sha256=sha256_bytes(SOURCE_TEXT.encode()),
        expected_patch_sha256=sha256_bytes(PATCH_TEXT.encode()),
        expected_patched_sha256=sha256_bytes(PATCHED_TEXT.encode()),
    )

    patched = overlay_root / "megatron/core/optimizer/distrib_optimizer.py"
    assert patched.read_text() == PATCHED_TEXT
    assert (source_root / "megatron/core/optimizer/distrib_optimizer.py").read_text() == SOURCE_TEXT
    receipt = json.loads(receipt_path.read_text())
    assert receipt == {
        "overlay_root": str(overlay_root.resolve()),
        "patch_path": str(patch_path.resolve()),
        "patch_sha256": sha256_bytes(PATCH_TEXT.encode()),
        "patched_file": "megatron/core/optimizer/distrib_optimizer.py",
        "patched_sha256": sha256_bytes(PATCHED_TEXT.encode()),
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "source_sha256": sha256_bytes(SOURCE_TEXT.encode()),
        "status": "applied",
    }


def test_prepare_overlay_is_idempotent_for_identical_verified_overlay(tmp_path: Path) -> None:
    preparer = load_preparer()
    source_root, overlay_root, patch_path, receipt_path = prepare_fixture(preparer, tmp_path)
    original_receipt = receipt_path.read_bytes()

    repeated_receipt = preparer.prepare_overlay(
        source_root=source_root,
        overlay_root=overlay_root,
        patch_path=patch_path,
        expected_source_sha256=sha256_bytes(SOURCE_TEXT.encode()),
        expected_patch_sha256=sha256_bytes(PATCH_TEXT.encode()),
        expected_patched_sha256=sha256_bytes(PATCHED_TEXT.encode()),
    )

    assert repeated_receipt == receipt_path
    assert receipt_path.read_bytes() == original_receipt


@pytest.mark.parametrize("drifted_input", ["source", "patch"])
def test_prepare_overlay_rejects_input_sha_drift_before_creating_overlay(
    tmp_path: Path, drifted_input: str
) -> None:
    preparer = load_preparer()
    source_root, overlay_root, patch_path = write_fixture(tmp_path)
    if drifted_input == "source":
        source_file = source_root / "megatron/core/optimizer/distrib_optimizer.py"
        source_file.write_text(SOURCE_TEXT + "# drift\n")
    else:
        patch_path.write_text(PATCH_TEXT + "# drift\n")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        preparer.prepare_overlay(
            source_root=source_root,
            overlay_root=overlay_root,
            patch_path=patch_path,
            expected_source_sha256=sha256_bytes(SOURCE_TEXT.encode()),
            expected_patch_sha256=sha256_bytes(PATCH_TEXT.encode()),
            expected_patched_sha256=sha256_bytes(PATCHED_TEXT.encode()),
        )

    assert not overlay_root.exists()


def test_prepare_overlay_rejects_unexpected_patched_hash_without_publishing(
    tmp_path: Path,
) -> None:
    preparer = load_preparer()
    source_root, overlay_root, patch_path = write_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="patched MCore file SHA256 mismatch"):
        preparer.prepare_overlay(
            source_root=source_root,
            overlay_root=overlay_root,
            patch_path=patch_path,
            expected_source_sha256=sha256_bytes(SOURCE_TEXT.encode()),
            expected_patch_sha256=sha256_bytes(PATCH_TEXT.encode()),
            expected_patched_sha256="0" * 64,
        )

    assert not overlay_root.exists()


@pytest.mark.parametrize("corruption", ["receipt", "patched_file"])
def test_prepare_overlay_never_clobbers_a_drifted_existing_overlay(
    tmp_path: Path, corruption: str
) -> None:
    preparer = load_preparer()
    source_root, overlay_root, patch_path, receipt_path = prepare_fixture(preparer, tmp_path)
    corrupted_path = (
        receipt_path
        if corruption == "receipt"
        else overlay_root / "megatron/core/optimizer/distrib_optimizer.py"
    )
    corrupted_path.write_text("drift\n")
    before = corrupted_path.read_bytes()

    with pytest.raises(RuntimeError):
        preparer.prepare_overlay(
            source_root=source_root,
            overlay_root=overlay_root,
            patch_path=patch_path,
            expected_source_sha256=sha256_bytes(SOURCE_TEXT.encode()),
            expected_patch_sha256=sha256_bytes(PATCH_TEXT.encode()),
            expected_patched_sha256=sha256_bytes(PATCHED_TEXT.encode()),
        )

    assert corrupted_path.read_bytes() == before


def test_prepare_overlay_rejects_symlinked_receipt(tmp_path: Path) -> None:
    preparer = load_preparer()
    source_root, overlay_root, patch_path, receipt_path = prepare_fixture(preparer, tmp_path)
    external_receipt = tmp_path / "external-receipt.json"
    receipt_path.rename(external_receipt)
    os.symlink(external_receipt, receipt_path)

    with pytest.raises(RuntimeError, match="valid receipt"):
        preparer.prepare_overlay(
            source_root=source_root,
            overlay_root=overlay_root,
            patch_path=patch_path,
            expected_source_sha256=sha256_bytes(SOURCE_TEXT.encode()),
            expected_patch_sha256=sha256_bytes(PATCH_TEXT.encode()),
            expected_patched_sha256=sha256_bytes(PATCHED_TEXT.encode()),
        )


def test_production_patch_is_bound_and_applies_the_lazy_state_fix(tmp_path: Path) -> None:
    preparer = load_preparer()
    source_root, overlay_root, _ = write_fixture(tmp_path)
    source_file = source_root / "megatron/core/optimizer/distrib_optimizer.py"
    source_file.write_text(PRODUCTION_SOURCE_TEXT)

    preparer.prepare_overlay(
        source_root=source_root,
        overlay_root=overlay_root,
        patch_path=PRODUCTION_PATCH,
        expected_source_sha256=sha256_bytes(PRODUCTION_SOURCE_TEXT.encode()),
        expected_patched_sha256=sha256_bytes(PRODUCTION_PATCHED_TEXT.encode()),
    )

    assert (
        overlay_root / "megatron/core/optimizer/distrib_optimizer.py"
    ).read_text() == PRODUCTION_PATCHED_TEXT


def test_cli_accepts_required_runtime_paths_and_fails_closed_on_unpinned_source(
    tmp_path: Path,
) -> None:
    source_root, overlay_root, _ = write_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PREPARER),
            "--source-root",
            str(source_root),
            "--overlay-root",
            str(overlay_root),
            "--patch",
            str(PRODUCTION_PATCH),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "source MCore file SHA256 mismatch" in result.stderr
    assert not overlay_root.exists()
