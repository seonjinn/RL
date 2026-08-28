from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "experiments" / "vllm_028_nemotron_bf16_matrix"
PINNED_BASE = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
PATCHED_IMAGE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
    "vllm-openai-v0.28.0-mrv2-dynamick-core-aarch64-ubuntu2404.sqsh"
)


def load_module(name: str) -> ModuleType:
    path = PACKAGE_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def make_patch_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.invalid")
    tracked = source / "value.txt"
    tracked.write_text("before\n", encoding="utf-8")
    git(source, "add", "value.txt")
    git(source, "commit", "-qm", "base")
    base_commit = git(source, "rev-parse", "HEAD")

    tracked.write_text("after\n", encoding="utf-8")
    git(source, "commit", "-qam", "change value")
    patch_commit = git(source, "rev-parse", "HEAD")
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    patch_path = patch_dir / "0001-change-value.patch"
    patch_path.write_text(git(source, "format-patch", "-1", "--stdout") + "\n")
    git(source, "reset", "--hard", base_commit)
    return source, patch_path, base_commit, patch_commit


def test_apply_patch_stack_applies_verified_series_once(tmp_path: Path) -> None:
    patchset = load_module("mrv2_patchset")
    source, patch_path, base_commit, patch_commit = make_patch_fixture(tmp_path)
    manifest = {
        "schema_version": 1,
        "vllm_version": "fixture",
        "base_commit": base_commit,
        "patches": [
            {
                "file": patch_path.name,
                "commit": patch_commit,
                "sha256": patchset.sha256_file(patch_path),
                "source": "fixture",
            }
        ],
    }

    applied = patchset.apply_patch_stack(
        source_root=source,
        patch_dir=patch_path.parent,
        manifest=manifest,
    )

    assert (source / "value.txt").read_text(encoding="utf-8") == "after\n"
    assert applied == [patch_commit]
    assert git(source, "status", "--short") == ""
    assert patchset.apply_patch_stack(
        source_root=source,
        patch_dir=patch_path.parent,
        manifest=manifest,
    ) == []


def test_apply_patch_stack_resolves_relative_patch_dir_before_git_am(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patchset = load_module("mrv2_patchset")
    source, patch_path, base_commit, patch_commit = make_patch_fixture(tmp_path)
    manifest = {
        "schema_version": 1,
        "vllm_version": "fixture",
        "base_commit": base_commit,
        "patches": [
            {
                "file": patch_path.name,
                "commit": patch_commit,
                "sha256": patchset.sha256_file(patch_path),
                "source": "fixture",
            }
        ],
    }
    monkeypatch.chdir(tmp_path)

    applied = patchset.apply_patch_stack(
        source_root=source,
        patch_dir=Path("patches"),
        manifest=manifest,
    )

    assert applied == [patch_commit]
    assert (source / "value.txt").read_text(encoding="utf-8") == "after\n"


def test_apply_patch_stack_idempotency_does_not_depend_on_mail_subject_cleanup(
    tmp_path: Path,
) -> None:
    patchset = load_module("mrv2_patchset")
    source, patch_path, base_commit, patch_commit = make_patch_fixture(tmp_path)
    patch_text = patch_path.read_text(encoding="utf-8").replace(
        "Subject: [PATCH] change value",
        "Subject: [PATCH] [Bugfix][MRV2] change value",
    )
    patch_path.write_text(patch_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "vllm_version": "fixture",
        "base_commit": base_commit,
        "patches": [
            {
                "file": patch_path.name,
                "commit": patch_commit,
                "sha256": patchset.sha256_file(patch_path),
                "source": "fixture",
            }
        ],
    }

    assert patchset.apply_patch_stack(
        source_root=source,
        patch_dir=patch_path.parent,
        manifest=manifest,
    ) == [patch_commit]
    assert git(source, "log", "-1", "--format=%s") == "change value"
    assert patchset.apply_patch_stack(
        source_root=source,
        patch_dir=patch_path.parent,
        manifest=manifest,
    ) == []


@pytest.mark.parametrize("mutation", ["wrong_base", "wrong_digest", "dirty_tree"])
def test_apply_patch_stack_fails_closed_before_mutating_source(
    tmp_path: Path, mutation: str
) -> None:
    patchset = load_module("mrv2_patchset")
    source, patch_path, base_commit, patch_commit = make_patch_fixture(tmp_path)
    manifest = {
        "schema_version": 1,
        "vllm_version": "fixture",
        "base_commit": base_commit,
        "patches": [
            {
                "file": patch_path.name,
                "commit": patch_commit,
                "sha256": patchset.sha256_file(patch_path),
                "source": "fixture",
            }
        ],
    }
    if mutation == "wrong_base":
        manifest["base_commit"] = "0" * 40
    elif mutation == "wrong_digest":
        manifest["patches"][0]["sha256"] = "0" * 64
    else:
        (source / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(patchset.PatchValidationError):
        patchset.apply_patch_stack(
            source_root=source,
            patch_dir=patch_path.parent,
            manifest=manifest,
        )

    assert (source / "value.txt").read_text(encoding="utf-8") == "before\n"
    assert git(source, "rev-parse", "HEAD") == base_commit


def test_core_manifest_pins_required_prs_and_local_mamba_fix() -> None:
    patchset = load_module("mrv2_patchset")

    manifest = patchset.load_manifest(PACKAGE_ROOT / "mrv2_patch_manifest.json")

    assert manifest["base_commit"] == PINNED_BASE
    assert [row["commit"] for row in manifest["patches"]] == [
        "9d531edf29553b37ce909db6cc4732eb045754d0",
        "2e5a430dc55409d1c82a2de41d98ca79fba189f3",
        "9e98959f0cf177a81259d85250803f98e86953f3",
        "4d9b59acd5213a93c7389a9a1c7195c73f1f5a89",
        "16b980a4e92f469cd209d0bbd84399cfae6fe3f4",
    ]
    assert all(
        patchset.sha256_file(PACKAGE_ROOT / "patches" / row["file"])
        == row["sha256"]
        for row in manifest["patches"]
    )


def test_canary_contract_exercises_every_runtime_k_in_one_engine() -> None:
    canary = load_module("benchmark_mrv2_patch_canary")

    assert canary.CANARY_BATCH_SIZES == (1, 2, 4, 8, 16)
    assert canary.CANARY_DYNAMIC_SCHEDULE == (
        "1:1:5,2:2:3,3:4:2,5:8:1,9:512:0"
    )
    assert canary.expected_k_by_batch() == {1: 5, 2: 3, 4: 2, 8: 1, 16: 0}


def test_validate_canary_payload_requires_exact_tokens_and_observed_draft_widths() -> None:
    canary = load_module("benchmark_mrv2_patch_canary")
    expected_k = {1: 5, 2: 3, 4: 2, 8: 1, 16: 0}
    rows: list[dict[str, Any]] = []
    for batch_size, k in expected_k.items():
        drafts = 20 if k else 0
        rows.append(
            {
                "bs": batch_size,
                "output_tokens": batch_size * 128,
                "spec_decode_metrics": {
                    "num_drafts": drafts,
                    "num_draft_tokens": drafts * k,
                },
            }
        )
    payload = {"status": "complete", "results": rows}

    validated = canary.validate_canary_payload(payload, osl=128)

    assert [row["observed_mean_draft_width"] for row in validated["results"]] == [
        5.0,
        3.0,
        2.0,
        1.0,
        0.0,
    ]
    assert all(row["tokens_ok"] for row in validated["results"])


def test_validate_canary_payload_rejects_max_k_work_hidden_under_reduced_k() -> None:
    canary = load_module("benchmark_mrv2_patch_canary")
    payload = {
        "status": "complete",
        "results": [
            {
                "bs": batch_size,
                "output_tokens": batch_size * 128,
                "spec_decode_metrics": {
                    "num_drafts": 20 if batch_size != 16 else 0,
                    "num_draft_tokens": 100 if batch_size != 16 else 0,
                },
            }
            for batch_size in (1, 2, 4, 8, 16)
        ],
    }

    with pytest.raises(ValueError, match="draft width"):
        canary.validate_canary_payload(payload, osl=128)


def test_render_canary_uses_patched_image_full_graph_and_patch_provenance() -> None:
    submit = load_module("submit_mrv2_patch_canary")
    manifest = json.loads(
        (PACKAGE_ROOT / "mrv2_patch_manifest.json").read_text(encoding="utf-8")
    )

    script = submit.render_canary_sbatch(
        model_key="super",
        experiment_dir=PACKAGE_ROOT,
        result_dir=Path("/lustre/results/patched-super"),
        manifest=manifest,
    )

    assert f"readonly STABLE_CONTAINER_IMAGE={PATCHED_IMAGE}" in script
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in script
    assert "FULL_AND_PIECEWISE" in script
    assert "--batch-sizes 1 2 4 8 16" in script
    assert "--dynamic-schedule 1:1:5,2:2:3,3:4:2,5:8:1,9:512:0" in script
    assert "mrv2_patch_manifest.json" in script
    assert "--patchset-manifest-sha256" in script
    assert "vllm-openai-v0.28.0-aarch64-ubuntu2404.sqsh" not in script


def test_render_canary_preserves_ultra_two_node_ray_topology() -> None:
    submit = load_module("submit_mrv2_patch_canary")
    manifest = json.loads(
        (PACKAGE_ROOT / "mrv2_patch_manifest.json").read_text(encoding="utf-8")
    )

    script = submit.render_canary_sbatch(
        model_key="ultra",
        experiment_dir=PACKAGE_ROOT,
        result_dir=Path("/lustre/results/patched-ultra"),
        manifest=manifest,
    )

    assert "#SBATCH --nodes=2" in script
    assert "--tensor-parallel-size 8" in script
    assert "--enable-expert-parallel" in script
    assert "--distributed-executor-backend ray" in script
    assert "/workspace/exp/run_multinode_ray.sh" in script


def test_stage_script_sets_local_git_identity_before_git_am() -> None:
    script = (
        PACKAGE_ROOT / "stage_vllm028_mrv2_patched_container.sbatch"
    ).read_text(encoding="utf-8")

    name_config = 'git -C "${VLLM_SOURCE}" config user.name'
    email_config = 'git -C "${VLLM_SOURCE}" config user.email'
    patch_call = 'python3 "${PATCHER}"'
    assert name_config in script
    assert email_config in script
    assert script.index(name_config) < script.index(patch_call)
    assert script.index(email_config) < script.index(patch_call)
