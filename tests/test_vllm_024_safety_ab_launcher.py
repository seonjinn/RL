from __future__ import annotations

import os
import hashlib
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_safety_ab_small_model.sh"
)


def _run_launcher(*args: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _create_pushed_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _run_git(repo, "config", "user.email", "launcher-test@nvidia.com")
    _run_git(repo, "config", "user.name", "Launcher Test")
    recipe_dir = repo / "examples" / "configs" / "recipes" / "llm" / "performance"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "grpo-llama3.1-8b-instruct-2n4g.yaml").write_text(
        "defaults: ../../../grpo_math_1B.yaml\n",
        encoding="utf-8",
    )
    (repo / "ray.sub").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "test: seed launcher repo")

    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    _run_git(repo, "remote", "add", "origin", str(remote))
    _run_git(repo, "push", "--set-upstream", "origin", "main")
    return repo, _run_git(repo, "rev-parse", "HEAD")


def test_dry_run_renders_matched_sync_jobs_from_repo_dirs() -> None:
    project = "nemorl-vllm024-safety-ab-contract"
    result = _run_launcher(
        "dry-run",
        "all",
        "sync",
        "10",
        CONTROL_REPO_DIR=str(REPO_ROOT),
        CANDIDATE_REPO_DIR=str(REPO_ROOT),
        EXPERIMENT_ROOT="/lustre/test/safety-ab",
        RUN_TAG="contract",
        WANDB_PROJECT=project,
        CONTAINER="/lustre/test/nemo-rl.sqsh",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    short_head = _git_head()[:12]
    recipe = (
        "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n4g.yaml"
    )

    assert output.count("[DRY-RUN] job ") == 2
    assert recipe in output
    assert "grpo.max_num_steps=10" in output
    assert "checkpointing.enabled=false" in output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in output
    assert "/opt/nemo_rl_venv/bin/python" in output
    assert "vllm.__version__" in output
    assert "0.24." in output
    assert "NRL_FORCE_REBUILD_VENVS" not in output
    assert "NEMO_RL_VENV_DIR" not in output
    assert "NEMO_RL_AB_" not in output
    assert "uv run" not in output
    assert "moe_backend" not in output
    assert "logger.wandb_enabled" not in output
    assert "logger.tensorboard_enabled" not in output
    assert "--nodes=2" in output
    assert "--segment=2" in output
    assert "--gres" not in output
    assert f"logger.wandb.project={project}" in output
    assert (
        f"logger.wandb.name=contract-control-sync-step10-val-recipe-r1-{short_head}"
        in output
    )
    assert (
        f"logger.wandb.name=contract-candidate-sync-step10-val-recipe-r1-{short_head}"
        in output
    )
    assert "/lustre/test/safety-ab/control/sync/step10" in output
    assert "/lustre/test/safety-ab/candidate/sync/step10" in output


def test_dry_run_renders_sync_async_and_step_matrix() -> None:
    result = _run_launcher(
        "dry-run",
        "all",
        "all",
        "all",
        CONTROL_REPO_DIR=str(REPO_ROOT),
        CANDIDATE_REPO_DIR=str(REPO_ROOT),
        EXPERIMENT_ROOT="/lustre/test/safety-ab",
        RUN_TAG="matrix",
        CONTAINER="/lustre/test/nemo-rl.sqsh",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    short_head = _git_head()[:12]

    assert output.count("[DRY-RUN] job ") == 12
    assert "grpo-llama3.1-8b-instruct-2n4g.yaml" in output
    assert "grpo-llama3.1-8b-instruct-2n4g-async-1off.yaml" in output
    assert "grpo.max_num_steps=10" in output
    assert "grpo.max_num_steps=20" in output
    assert "grpo.max_num_steps=40" in output
    for cohort in ("control", "candidate"):
        for recipe in ("sync", "async-1off"):
            for steps in (10, 20, 40):
                assert (
                    f"matrix-{cohort}-{recipe}-step{steps}-val-recipe-r1-{short_head}"
                    in output
                )


def test_dry_run_resolves_commit_to_deterministic_lustre_checkout() -> None:
    commit = _git_head()
    result = _run_launcher(
        "dry-run",
        "control",
        "async-1off",
        "20",
        SOURCE_REPO_DIR=str(REPO_ROOT),
        CONTROL_COMMIT="HEAD",
        CHECKOUT_ROOT="/lustre/test/checkouts",
        EXPERIMENT_ROOT="/lustre/test/safety-ab",
        RUN_TAG="commit-contract",
        CONTAINER="/lustre/test/nemo-rl.sqsh",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    checkout = f"/lustre/test/checkouts/control-{commit[:12]}"

    assert f"CONTAINER_WORKDIR={checkout}" in output
    assert f"PYTHONPATH={checkout}" in output
    assert f"[DRY-RUN] provenance control commit={commit} repo={checkout}" in output
    assert (
        f"commit-contract-control-async-1off-step20-val-recipe-r1-{commit[:12]}"
        in output
    )


def test_dry_run_supports_validation_free_replicated_steady_state() -> None:
    result = _run_launcher(
        "dry-run",
        "all",
        "async-1off",
        "40",
        CONTROL_REPO_DIR=str(REPO_ROOT),
        CANDIDATE_REPO_DIR=str(REPO_ROOT),
        EXPERIMENT_ROOT="/lustre/test/safety-ab",
        RUN_TAG="steady",
        VALIDATION_MODE="off",
        REPLICATES="3",
        CONTAINER="/lustre/test/nemo-rl.sqsh",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert output.count("[DRY-RUN] job ") == 6
    assert "grpo.val_period=0" in output
    assert "grpo.val_at_start=false" in output
    assert "grpo.val_at_end=false" in output
    for cohort in ("control", "candidate"):
        for replicate in (1, 2, 3):
            assert f"steady-{cohort}-async-1off-step40-val-off-r{replicate}-" in output


def test_test_only_invokes_lyris_scheduler_validation(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf 'sbatch-args'\nprintf ' %q' \"$@\"\nprintf '\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    experiment_root = tmp_path / "runs"

    result = _run_launcher(
        "test-only",
        "control",
        "async-1off",
        "20",
        CONTROL_REPO_DIR=str(REPO_ROOT),
        EXPERIMENT_ROOT=str(experiment_root),
        RUN_TAG="test-only-contract",
        CONTAINER=str(container),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "sbatch-args --test-only" in output
    assert "--nodes=2" in output
    assert "--segment=2" in output
    assert "--gres" not in output
    assert (experiment_root / "control" / "async-1off").is_dir()


def test_test_only_rejects_mismatched_recipe_blobs(tmp_path: Path) -> None:
    control_root = tmp_path / "control"
    candidate_root = tmp_path / "candidate"
    control_root.mkdir()
    candidate_root.mkdir()
    control_repo, _control_commit = _create_pushed_repo(control_root)
    candidate_repo, _candidate_commit = _create_pushed_repo(candidate_root)
    candidate_recipe = (
        candidate_repo
        / "examples"
        / "configs"
        / "recipes"
        / "llm"
        / "performance"
        / "grpo-llama3.1-8b-instruct-2n4g.yaml"
    )
    candidate_recipe.write_text("defaults: changed.yaml\n", encoding="utf-8")
    _run_git(candidate_repo, "add", str(candidate_recipe))
    _run_git(candidate_repo, "commit", "-m", "test: change recipe")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()

    result = _run_launcher(
        "test-only",
        "all",
        "sync",
        "10",
        CONTROL_REPO_DIR=str(control_repo),
        CANDIDATE_REPO_DIR=str(candidate_repo),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        CONTAINER=str(container),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 2
    assert "recipe blob mismatch" in result.stderr


def test_submit_records_reproducible_manifest(tmp_path: Path) -> None:
    repo, commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >"$SBATCH_ARGS_FILE"\n'
        "printf '4242\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.write_text("container-contract\n", encoding="utf-8")
    experiment_root = tmp_path / "runs"
    sbatch_args_file = tmp_path / "sbatch-args.txt"

    result = _run_launcher(
        "submit",
        "control",
        "sync",
        "10",
        SOURCE_REPO_DIR=str(repo),
        CONTROL_COMMIT=commit,
        CHECKOUT_ROOT=str(tmp_path / "checkouts"),
        EXPERIMENT_ROOT=str(experiment_root),
        RUN_TAG="submit-contract",
        WANDB_PROJECT="shared-safety-ab",
        WANDB_API_KEY="test-only-key",
        CONTAINER=str(container),
        SBATCH_ARGS_FILE=str(sbatch_args_file),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    assert sbatch_args_file.read_text(encoding="utf-8").startswith("--parsable ")
    manifest = (experiment_root / "submissions.tsv").read_text(encoding="utf-8")
    assert "cohort\trecipe\tsteps\tvalidation\treplicate\tjob_id\tcommit" in manifest
    assert f"\tcontrol\tsync\t10\trecipe\t1\t4242\t{commit}\t" in manifest
    assert "shared-safety-ab" in manifest
    assert "submit-contract-control-sync-step10-val-recipe-r1" in manifest
    assert "grpo-llama3.1-8b-instruct-2n4g.yaml" in manifest
    expected_container_sha256 = hashlib.sha256(container.read_bytes()).hexdigest()
    assert expected_container_sha256 in manifest
    assert "/opt/nemo_rl_venv/bin/python" in manifest
    assert "test-only-key" not in manifest


def test_submit_rejects_untracked_checkout_content(tmp_path: Path) -> None:
    repo, commit = _create_pushed_repo(tmp_path)
    checkout_root = tmp_path / "checkouts"
    checkout_root.mkdir()
    checkout = checkout_root / f"control-{commit[:12]}"
    _run_git(repo, "worktree", "add", "--detach", str(checkout), commit)
    (checkout / "untracked_override.py").write_text("VALUE = 1\n", encoding="utf-8")
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()

    result = _run_launcher(
        "submit",
        "control",
        "sync",
        "10",
        SOURCE_REPO_DIR=str(repo),
        CONTROL_COMMIT=commit,
        CHECKOUT_ROOT=str(checkout_root),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
        CONTAINER=str(container),
    )

    assert result.returncode == 2
    assert "clean checkout" in result.stderr


def test_submit_rejects_mutable_repo_dir(tmp_path: Path) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()

    result = _run_launcher(
        "submit",
        "control",
        "sync",
        "10",
        CONTROL_REPO_DIR=str(repo),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
        CONTAINER=str(container),
    )

    assert result.returncode == 2
    assert "CONTROL_COMMIT, not a mutable CONTROL_REPO_DIR" in result.stderr


def test_submit_rejects_incorrect_container_sha256(tmp_path: Path) -> None:
    repo, commit = _create_pushed_repo(tmp_path)
    container = tmp_path / "nemo-rl.sqsh"
    container.write_text("container-contract\n", encoding="utf-8")

    result = _run_launcher(
        "submit",
        "control",
        "sync",
        "10",
        SOURCE_REPO_DIR=str(repo),
        CONTROL_COMMIT=commit,
        CHECKOUT_ROOT=str(tmp_path / "checkouts"),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
        CONTAINER=str(container),
        CONTAINER_SHA256="incorrect",
    )

    assert result.returncode == 2
    assert "container SHA256 mismatch" in result.stderr
