"""Dry-run contracts for the MXFP8 MoE tactic audit launchers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
    assert "CACHE_ROOT=" not in (AUDIT_DIR / "submit_trace_ptyche.sh").read_text(
        encoding="ascii"
    )


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
    assert "nsys stats --report nvtxppsum" in output
    assert "nsys_to_component_csv.py" in output
    assert "--stock-cache" in output
    assert "stock_input_cache_root=" in output
    assert "mkdir -p ${RUN_ROOT} ${CACHE_ROOT}" not in output


def test_launchers_resolve_hf_cache_roots_to_exact_snapshots() -> None:
    """Catch a launcher passing an HF cache root to its runtime as a model."""
    for launcher_name in (
        "submit_trace_ptyche.sh",
        "submit_shmoo_ptyche.sh",
        "submit_validation_ptyche.sh",
    ):
        source = (AUDIT_DIR / launcher_name).read_text(encoding="ascii")
        assert "audit_resolve_model_snapshot" in source
        assert "HF_MODEL_CACHE_DIR" in source
        assert "MODEL_SNAPSHOT" in source


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
    assert "--write-run-evidence" in candidate_output
    assert "--write-run-evidence" in candidate_output
    assert "train_data_step" in (AUDIT_DIR / "collect_results.py").read_text(
        encoding="ascii"
    )
    assert '"realized_generated_tokens": None' not in candidate_output
    assert "RUNTIME_FINGERPRINTS_JSON" in candidate_output
    assert "\nPY\nif [[ 8 -eq 2 ]]" not in candidate_output
    assert "trap" in candidate_output
    assert "policy.model_name=" in candidate_output
    assert "HF_HUB_OFFLINE=1" in candidate_output
    assert "TRANSFORMERS_OFFLINE=1" in candidate_output
    assert "/validation/stock/" not in candidate_output
    assert "--return-tokens-as-token-ids" in candidate_output
    assert '"return_token_ids": True' in candidate_output
    assert 'choice.get("token_ids")' in candidate_output
    assert "ord(char)" not in candidate_output


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

    for action in ("test-only", "submit"):
        rejected = subprocess.run(
            ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
            cwd=REPO_ROOT,
            env=os.environ
            | {
                "ACTION": action,
                "VALIDATION_MODE": "compare",
                "WORK_ROOT": str(tmp_path),
            },
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        assert "is local" in rejected.stderr

    action_run = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=os.environ | {"ACTION": "run", "WORK_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert action_run.returncode == 2
    assert "Unsupported ACTION: run" in action_run.stderr

    bin_dir = tmp_path / "compare-bin"
    bin_dir.mkdir()
    compare_log = tmp_path / "compare.log"
    (bin_dir / "python").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$COMPARE_LOG"\n',
        encoding="ascii",
    )
    (bin_dir / "python").chmod(0o755)
    local_run = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=os.environ
        | {
            "ACTION": "dry-run",
            "COMPARE_ACTION": "run",
            "VALIDATION_MODE": "compare",
            "WORK_ROOT": str(tmp_path),
            "COMPARE_LOG": str(compare_log),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    compare_calls = compare_log.read_text(encoding="ascii").splitlines()
    assert len(compare_calls) == 2
    assert "validate_correctness.py generation" in compare_calls[0]
    assert "compare_gsm8k.py" in compare_calls[1]


def _make_model_cache(tmp_path: Path, *, symlinked: bool = False) -> Path:
    """Create the smallest complete Qwen3 snapshot accepted by launchers."""
    model_cache = tmp_path / "hf/hub/models--Qwen--Qwen3-30B-A3B"
    snapshot = model_cache / "snapshots/revision"
    (model_cache / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (model_cache / "refs/main").write_text("revision\n", encoding="ascii")
    source = tmp_path / "blobs"
    source.mkdir()
    index = snapshot / "model.safetensors.index.json"
    if symlinked:
        (source / "index").write_text("{}\n", encoding="ascii")
        index.symlink_to(source / "index")
    else:
        index.write_text("{}\n", encoding="ascii")
    for index in range(16):
        shard = snapshot / f"model-{index:05d}.safetensors"
        if symlinked:
            blob = source / f"shard-{index:05d}"
            blob.write_text("weight\n", encoding="ascii")
            shard.symlink_to(blob)
        else:
            shard.write_text("weight\n", encoding="ascii")
    return model_cache


def test_validation_test_only_rejects_a_missing_cache_before_sbatch(
    tmp_path: Path,
) -> None:
    """Catch cache creation or scheduler calls after a missing-cache preflight."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    (bin_dir / "sbatch").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$SBATCH_LOG"\n',
        encoding="ascii",
    )
    (bin_dir / "sbatch").chmod(0o755)
    model_cache = _make_model_cache(tmp_path)
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


def test_shmoo_test_only_rejects_a_missing_stock_input_cache_before_sbatch(
    tmp_path: Path,
) -> None:
    """Catch a shmoo job creating its immutable stock input cache."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    (bin_dir / "sbatch").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$SBATCH_LOG"\n',
        encoding="ascii",
    )
    (bin_dir / "sbatch").chmod(0o755)
    model_cache = _make_model_cache(tmp_path)
    container = tmp_path / "container.sqsh"
    profiles = tmp_path / "selected_profiles.json"
    container.write_text("container\n", encoding="ascii")
    profiles.write_text("[]\n", encoding="ascii")
    result = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_shmoo_ptyche.sh")],
        cwd=REPO_ROOT,
        env=os.environ
        | {
            "ACTION": "test-only",
            "RUN_ID": "stock-cache-missing",
            "WORK_ROOT": str(tmp_path),
            "REPO_DIR_OVERRIDE": str(REPO_ROOT),
            "CUSTOM_VLLM_ROOT": str(tmp_path / "vllm"),
            "CONTAINER": str(container),
            "HF_MODEL_CACHE_DIR": str(model_cache),
            "SELECTED_PROFILES": str(profiles),
            "STOCK_INPUT_CACHE_ROOT": str(tmp_path / "missing-stock-cache"),
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
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], check=True, cwd=path
    )
    subprocess.run(["git", "config", "user.name", "Test User"], check=True, cwd=path)
    (path / "tracked.txt").write_text("clean\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], check=True, cwd=path)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], check=True, cwd=path)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def test_provenance_rejects_dirty_tracked_source(tmp_path: Path) -> None:
    """Catch submit preflight silently accepting tracked source modifications."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "tracked.txt").write_text("dirty\n", encoding="ascii")

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; audit_assert_clean_tracked "$2"',
            "bash",
            str(PROVENANCE),
            str(repo),
        ],
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
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "linked", str(linked)],
        check=True,
        cwd=source,
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; audit_assert_clean_tracked "$2"',
            "bash",
            str(PROVENANCE),
            str(linked),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""


def test_provenance_hashes_hf_symlink_snapshots_and_rejects_broken_links(
    tmp_path: Path,
) -> None:
    """Catch cache hashing that ignores Hub blob links or follows broken inputs."""
    model_cache = _make_model_cache(tmp_path, symlinked=True)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; audit_resolve_model_snapshot "$2" 16; audit_sha256_path "$2/snapshots/revision"',
            "bash",
            str(PROVENANCE),
            str(model_cache),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert lines[0].endswith("/snapshots/revision\trevision")
    assert len(lines[1]) == 64

    (model_cache / "snapshots/revision/broken.safetensors").symlink_to(
        tmp_path / "blobs/missing"
    )
    broken = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; audit_sha256_path "$2/snapshots/revision"',
            "bash",
            str(PROVENANCE),
            str(model_cache),
        ],
        capture_output=True,
        text=True,
    )
    assert broken.returncode != 0
    assert "Broken symlink" in broken.stderr


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
            'source "$1"; audit_write_manifest "$2" trace "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"',
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


def _make_submit_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Build isolated repos and fake scheduler tools for submit-only launcher tests."""
    repo = tmp_path / "repo"
    vllm = tmp_path / "vllm"
    repo.mkdir()
    vllm.mkdir()
    _init_git_repo(repo)
    vllm_commit = _init_git_repo(vllm)
    recipe = repo / "examples/configs/recipes/llm/performance"
    recipe.mkdir(parents=True)
    (recipe / "grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml").write_text(
        "recipe: test\n", encoding="ascii"
    )
    subprocess.run(["git", "add", "."], check=True, cwd=repo)
    subprocess.run(["git", "commit", "-q", "-m", "recipe"], check=True, cwd=repo)
    model_cache = _make_model_cache(tmp_path)
    cache = tmp_path / "cache/candidate"
    cache.mkdir(parents=True)
    (cache / "entry").write_text("cache\n", encoding="ascii")
    (cache / "cache_manifest.json").write_text("{}\n", encoding="ascii")
    container = tmp_path / "container.sqsh"
    container.write_text("container\n", encoding="ascii")
    evaluator = tmp_path / "gsm8k.py"
    evaluator.write_text("print('not run')\n", encoding="ascii")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    order_log = tmp_path / "order.log"
    real_git = shutil.which("git")
    assert real_git is not None
    (bin_dir / "git").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-C" && "$3" == "pull" ]]; then printf "pull\\n" >> "$ORDER_LOG"; exit 0; fi\n'
        f'exec "{real_git}" "$@"\n',
        encoding="ascii",
    )
    (bin_dir / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        'test -f "$EXPECTED_MANIFEST"\n'
        'printf "sbatch\\n" >> "$ORDER_LOG"\n',
        encoding="ascii",
    )
    for executable in (bin_dir / "git", bin_dir / "sbatch"):
        executable.chmod(0o755)

    env = os.environ | {
        "ACTION": "submit",
        "ARM": "candidate",
        "MAX_STEPS": "2",
        "RUN_ID": "submit-test",
        "WORK_ROOT": str(tmp_path),
        "REPO_DIR_OVERRIDE": str(repo),
        "CUSTOM_VLLM_ROOT": str(vllm),
        "EXPECTED_VLLM_COMMIT": vllm_commit,
        "CONTAINER": str(container),
        "HF_MODEL_CACHE_DIR": str(model_cache),
        "CANDIDATE_CACHE_ROOT": str(cache),
        "GSM8K_EVALUATOR": str(evaluator),
        "ORDER_LOG": str(order_log),
        "EXPECTED_MANIFEST": str(
            tmp_path
            / "experiments/mxfp8-moe-tactic-audit/validation/candidate/submit-test/steps-2/run_manifest.json"
        ),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    return env, order_log, Path(env["EXPECTED_MANIFEST"])


def test_validation_submit_pulls_preflights_and_then_calls_fake_sbatch(
    tmp_path: Path,
) -> None:
    """Catch submit ordering, unresolved snapshots, or scheduler calls without a manifest."""
    env, order_log, manifest_path = _make_submit_environment(tmp_path)
    result = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert order_log.read_text(encoding="ascii").splitlines() == ["pull", "sbatch"]
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    assert manifest["model_snapshot_sha256"] != "dry-run-not-validated"
    assert manifest["cache_sha256"] != "dry-run-not-validated"
    assert "--constraint=GB200" in result.stdout
    assert "--segment=4" in result.stdout


def _make_valid_two_step_smoke(
    tmp_path: Path,
) -> tuple[dict[str, str], Path, Path, Path]:
    """Create a two-step manifest and its arm-local marker through fake submit."""
    env, order_log, manifest_path = _make_submit_environment(tmp_path)
    subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    marker_path = manifest_path.parent.parent / "smoke-candidate-valid.json"
    marker_path.write_text(
        json.dumps(
            {
                "cache_sha256": manifest["cache_sha256"],
                "model_snapshot_sha256": manifest["model_snapshot_sha256"],
                "smoke_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="ascii",
    )
    env.update(
        {
            "ACTION": "test-only",
            "MAX_STEPS": "8",
            "SMOKE_MANIFEST": str(manifest_path),
            "SMOKE_MARKER": str(marker_path),
        }
    )
    return env, order_log, manifest_path, marker_path


def test_eight_step_gate_accepts_a_matching_smoke_and_reaches_fake_sbatch(
    tmp_path: Path,
) -> None:
    """Catch a smoke gate that hashes a different input list than its manifest."""
    env, order_log, _, _ = _make_valid_two_step_smoke(tmp_path)
    result = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "grpo.max_num_steps=8" in result.stdout
    assert order_log.read_text(encoding="ascii").splitlines() == [
        "pull",
        "sbatch",
        "sbatch",
    ]


def test_eight_step_gate_rejects_a_mismatched_execution_inputs_hash(
    tmp_path: Path,
) -> None:
    """Catch a stale smoke whose exact execution input list no longer matches."""
    env, order_log, manifest_path, _ = _make_valid_two_step_smoke(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["execution_inputs_sha256"] = "mismatched-inputs"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Stale smoke manifest execution_inputs_sha256" in result.stderr
    assert order_log.read_text(encoding="ascii").splitlines() == ["pull", "sbatch"]


def test_validation_submit_rejects_an_existing_run_root_before_sbatch(
    tmp_path: Path,
) -> None:
    """Catch a submit that can overwrite a prior run's evidence."""
    env, order_log, manifest_path = _make_submit_environment(tmp_path)
    manifest_path.parent.mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Run root already exists" in result.stderr
    assert order_log.read_text(encoding="ascii").splitlines() == ["pull"]


def test_eight_step_gate_rejects_stale_source_or_recipe_smoke(
    tmp_path: Path,
) -> None:
    """Catch an eight-step arm accepting a smoke from changed source or recipe."""
    env, _, manifest_path = _make_submit_environment(tmp_path)
    env.update({"ACTION": "test-only", "MAX_STEPS": "8"})
    smoke_manifest = manifest_path.parent.parent / "steps-2/run_manifest.json"
    smoke_manifest.parent.mkdir(parents=True)
    repo = Path(env["REPO_DIR_OVERRIDE"])
    vllm = Path(env["CUSTOM_VLLM_ROOT"])
    current_nemo = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    stale = {
        "nemo_rl_commit": current_nemo,
        "vllm_commit": env["EXPECTED_VLLM_COMMIT"],
        "recipe_sha256": "stale-recipe",
    }
    smoke_manifest.write_text(json.dumps(stale), encoding="ascii")
    marker = smoke_manifest.parent.parent / "smoke-candidate-placeholder.json"
    marker.write_text("{}\n", encoding="ascii")
    env["SMOKE_MANIFEST"] = str(smoke_manifest)
    env["SMOKE_MARKER"] = str(marker)

    result = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Stale smoke manifest recipe_sha256" in result.stderr


def test_eight_step_gate_rejects_stale_source_smoke(tmp_path: Path) -> None:
    """Catch a smoke marker surviving a NeMo source change."""
    env, _, manifest_path = _make_submit_environment(tmp_path)
    env.update({"ACTION": "test-only", "MAX_STEPS": "8"})
    smoke_manifest = manifest_path.parent.parent / "steps-2/run_manifest.json"
    smoke_manifest.parent.mkdir(parents=True)
    smoke_manifest.write_text(
        json.dumps(
            {
                "nemo_rl_commit": "stale-source",
                "vllm_commit": env["EXPECTED_VLLM_COMMIT"],
            }
        ),
        encoding="ascii",
    )
    marker = smoke_manifest.parent.parent / "smoke-candidate-placeholder.json"
    marker.write_text("{}\n", encoding="ascii")
    env["SMOKE_MANIFEST"] = str(smoke_manifest)
    env["SMOKE_MARKER"] = str(marker)

    result = subprocess.run(
        ["bash", str(AUDIT_DIR / "submit_validation_ptyche.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Stale smoke manifest nemo_rl_commit" in result.stderr
