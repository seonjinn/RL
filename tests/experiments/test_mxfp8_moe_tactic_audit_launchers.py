"""Dry-run contracts for the MXFP8 MoE tactic audit launchers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO_ROOT / "experiments" / "mxfp8_moe_tactic_audit"
PROVENANCE = AUDIT_DIR / "provenance.sh"


def _dry_run(
    launcher_name: str, tmp_path: Path, extra_env: dict[str, str] | None = None
) -> str:
    """Run a launcher without submitting and return its rendered command."""
    launcher = AUDIT_DIR / launcher_name
    assert launcher.is_file(), f"missing launcher: {launcher}"

    env = os.environ | {
        "ACTION": "dry-run",
        "WORK_ROOT": str(tmp_path),
        "RUN_ID": "launcher-test",
        "REPO_DIR_OVERRIDE": str(REPO_ROOT),
        "CUSTOM_VLLM_ROOT": str(tmp_path / "vllm"),
        "CONTAINER": str(tmp_path / "nemo-rl.sqsh"),
    }
    if extra_env is not None:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(launcher)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_trace_dry_run_is_eager_and_metadata_only(tmp_path: Path) -> None:
    """Catch a trace launcher that can silently turn into a performance run."""
    output = _dry_run("submit_trace_ptyche.sh", tmp_path)

    assert "policy.generation.vllm_cfg.enforce_eager=true" in output
    assert "grpo.max_num_steps=2" in output
    assert "VLLM_MXFP8_MOE_TRACE_DIR=" in output
    assert "trace_is_metadata_only=true" in output
    assert "logger.wandb_enabled=false" in output
    assert "--constraint=GB200" in output
    assert "--segment=4" in output


def test_shmoo_dry_run_requests_one_gb200_for_five_hours(tmp_path: Path) -> None:
    """Catch resource or replay drift in the one-GPU tactic shmoo."""
    output = _dry_run("submit_shmoo_ptyche.sh", tmp_path)

    assert "--nodes=1" in output
    assert "--ntasks=1" in output
    assert "--time=05:00:00" in output
    assert "--constraint=GB200" in output
    assert "--warmups 3" in output
    assert "--repetitions 10" in output
    assert "CUDA Graph" in output
    assert "nsys profile" in output


def test_validation_dry_runs_keep_stock_and_candidate_isolated(tmp_path: Path) -> None:
    """Catch cache sharing or accidental scheduling coupling between arms."""
    stock_output = _dry_run(
        "submit_validation_ptyche.sh", tmp_path, {"ARM": "stock", "MAX_STEPS": "2"}
    )
    candidate_output = _dry_run(
        "submit_validation_ptyche.sh",
        tmp_path,
        {"ARM": "candidate", "MAX_STEPS": "8"},
    )

    assert "policy.generation.vllm_cfg.enforce_eager=false" in stock_output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in candidate_output
    assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=" in stock_output
    assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=" in candidate_output
    assert "/cache/stock" in stock_output
    assert "/cache/candidate" in candidate_output
    assert "grpo.max_num_steps=2" in stock_output
    assert "grpo.max_num_steps=8" in candidate_output
    assert "--dependency" not in stock_output
    assert "--dependency" not in candidate_output
    assert "--constraint=GB200" in candidate_output
    assert "--segment=4" in candidate_output
    assert "MXFP8_MOE_CUDA_GRAPH_REPLAY=required" in candidate_output
    assert "vllm serve" in candidate_output
    assert "generation.jsonl" in candidate_output
    assert "trap" in candidate_output
    assert "policy.model_name=" in candidate_output
    assert "HF_HUB_OFFLINE=1" in candidate_output
    assert "TRANSFORMERS_OFFLINE=1" in candidate_output
    assert "/validation/stock/" not in candidate_output


def test_compare_mode_is_the_only_cross_arm_validation_path(tmp_path: Path) -> None:
    """Catch run-mode validation reading artifacts from the other arm."""
    output = _dry_run(
        "submit_validation_ptyche.sh",
        tmp_path,
        {"ARM": "candidate", "MAX_STEPS": "8", "VALIDATION_MODE": "compare"},
    )

    assert "validate_correctness.py generation" in output
    assert "compare_gsm8k.py" in output
    assert "/validation/stock/" in output


def test_validation_test_only_rejects_a_missing_cache_before_sbatch(
    tmp_path: Path,
) -> None:
    """Catch cache creation or scheduler calls after a missing-cache preflight."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    (bin_dir / "sbatch").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$SBATCH_LOG\"\n",
        encoding="ascii",
    )
    (bin_dir / "sbatch").chmod(0o755)
    model_cache = tmp_path / "hf/hub/models--Qwen--Qwen3-30B-A3B"
    snapshot = model_cache / "snapshots/revision"
    (model_cache / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (model_cache / "refs/main").write_text("revision\n", encoding="ascii")
    (snapshot / "model.safetensors.index.json").write_text("{}\n", encoding="ascii")
    for index in range(16):
        (snapshot / f"model-{index:05d}.safetensors").write_text("weight\n", encoding="ascii")
    (tmp_path / "container.sqsh").write_text("container\n", encoding="ascii")
    result = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=os.environ
        | {
            "ACTION": "test-only",
            "ARM": "candidate",
            "RUN_ID": "cache-missing",
            "WORK_ROOT": str(tmp_path),
            "REPO_DIR_OVERRIDE": str(REPO_ROOT),
            "CUSTOM_VLLM_ROOT": str(tmp_path / "vllm"),
            "CONTAINER": str(tmp_path / "container.sqsh"),
            "HF_MODEL_CACHE_DIR": str(model_cache),
            "SBATCH_LOG": str(sbatch_log),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing or empty required cache" in result.stderr
    assert not sbatch_log.exists()


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], check=True, cwd=path)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, cwd=path)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True, cwd=path)
    (path / "tracked.txt").write_text("clean\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], check=True, cwd=path)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], check=True, cwd=path)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_provenance_rejects_dirty_tracked_source(tmp_path: Path) -> None:
    """Catch submit preflight silently accepting tracked source modifications."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "tracked.txt").write_text("dirty\n", encoding="ascii")

    result = subprocess.run(
        ["bash", "-c", 'source "$1"; audit_assert_clean_tracked "$2"', "bash", str(PROVENANCE), str(repo)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Tracked source is dirty" in result.stderr


def test_provenance_accepts_a_clean_linked_worktree(tmp_path: Path) -> None:
    """Catch preflight rejecting the .git file used by linked worktrees."""
    source = tmp_path / "source"
    source.mkdir()
    _init_git_repo(source)
    linked = tmp_path / "linked"
    subprocess.run(["git", "worktree", "add", "-q", "-b", "linked", str(linked)], check=True, cwd=source)

    result = subprocess.run(
        ["bash", "-c", 'source "$1"; audit_assert_clean_tracked "$2"', "bash", str(PROVENANCE), str(linked)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""


def test_provenance_manifest_hashes_inputs_without_environment_credentials(
    tmp_path: Path,
) -> None:
    """Catch a manifest that omits a fingerprint or leaks a credential value."""
    repo = tmp_path / "repo"
    vllm = tmp_path / "vllm"
    repo.mkdir()
    vllm.mkdir()
    _init_git_repo(repo)
    vllm_commit = _init_git_repo(vllm)
    recipe = repo / "recipe.yaml"
    container = tmp_path / "container.sqsh"
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    recipe.write_text("recipe: test\n", encoding="ascii")
    container.write_text("container\n", encoding="ascii")
    model.mkdir()
    cache.mkdir()
    (model / "weights.bin").write_text("weights\n", encoding="ascii")
    (cache / "entries.json").write_text("{}\n", encoding="ascii")

    secret = "do-not-write-this-credential"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; audit_write_manifest "$2" trace "$3" "$4" "$5" "$6" "$7" "$8" "$9" "$10"',
            "bash",
            str(PROVENANCE),
            str(output),
            str(repo),
            str(vllm),
            vllm_commit,
            str(container),
            "recipe.yaml",
            str(model),
            str(cache),
            str(AUDIT_DIR),
        ],
        check=True,
        env=os.environ | {"HF_TOKEN": secret, "WANDB_API_KEY": secret},
        capture_output=True,
        text=True,
    )
    manifest_text = (output / "run_manifest.json").read_text(encoding="ascii")
    manifest = json.loads(manifest_text)

    assert secret not in result.stdout
    assert secret not in result.stderr
    assert secret not in manifest_text
    assert manifest["run_kind"] == "trace"
    for field_name in (
        "cache_sha256",
        "container_sha256",
        "model_snapshot_sha256",
        "recipe_sha256",
        "scripts_sha256",
    ):
        assert len(manifest[field_name]) == 64
