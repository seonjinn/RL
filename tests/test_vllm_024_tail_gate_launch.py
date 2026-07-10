from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "vllm_024_upgrade"
    / "submit_tail_gated_specdec_step20.sh"
)
V1_VARIANTS = ("baseline_v1", "always_on_v1_k5", "stock_dynamic_v1")
V2_VARIANTS = (
    "baseline_v2",
    "always_on_v2_k5",
    "fastrl_threshold_v2_k5",
    "efficient_roofline_v2_k5",
)
GATED_V2_VARIANTS = ("fastrl_threshold_v2_k5", "efficient_roofline_v2_k5")


def _run_launcher(*args: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )


def _dry_run(model: str, variant: str) -> str:
    result = _run_launcher(
        "dry-run",
        model,
        variant,
        REPO_DIR="/lustre/test/nemo-rl",
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER="/lustre/test/nemo-rl.sqsh",
        EXPERIMENT_ROOT="/lustre/test/tail-gate-runs",
        RUN_TAG="contract",
        ATTEMPT_ID="attempt-1",
        ROOFLINE_CONFIG="/lustre/test/calibrations/qwen3.json",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_dry_run_exposes_exactly_the_seven_planned_variants() -> None:
    result = _run_launcher(
        "dry-run",
        "all",
        "all",
        REPO_DIR="/lustre/test/nemo-rl",
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER="/lustre/test/nemo-rl.sqsh",
        EXPERIMENT_ROOT="/lustre/test/tail-gate-runs",
        RUN_TAG="contract",
        ATTEMPT_ID="attempt-1",
        ROOFLINE_CONFIG="/lustre/test/calibrations/qwen3.json",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[DRY-RUN] job ") == 14
    for variant in (*V1_VARIANTS, *V2_VARIANTS):
        assert result.stdout.count(f"variant={variant}") == 2


def test_v1_variants_preserve_stock_runner_and_dynamic_contract() -> None:
    for variant in V1_VARIANTS:
        output = _dry_run("qwen30ba3b", variant)

        assert "VLLM_USE_V2_MODEL_RUNNER=0" in output
        assert "cudagraph_mode=PIECEWISE" in output
        assert "scheduler_cls=" not in output
        assert "sd_tail_gate_mode=" not in output
        assert "grpo-qwen3-30ba3b-4n4g.yaml" in output
        assert "--nodes=4" in output
        assert "--segment=4" in output

    dynamic_output = _dry_run("qwen30ba3b", "stock_dynamic_v1")
    assert (
        "num_speculative_tokens_per_batch_size=[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]"
        in dynamic_output.replace("\\", "")
    )


def test_v2_variants_preserve_runner_graph_and_gate_boundaries() -> None:
    for variant in V2_VARIANTS:
        output = _dry_run("qwen32b", variant)

        assert "VLLM_USE_V2_MODEL_RUNNER=1" in output
        assert "cudagraph_mode=FULL_AND_PIECEWISE" in output
        assert "grpo-qwen3-32b-4n4g.yaml" in output
        assert "--nodes=4" in output
        assert "--segment=4" in output

        if variant in GATED_V2_VARIANTS:
            assert (
                "scheduler_cls=nemo_rl.models.generation.vllm.tail_gate_scheduler.TailGatedScheduler"
                in output
            )
            assert "sd_tail_gate_mode=" in output
            assert "sd_tail_gate_threshold=32" in output
            assert "sd_tail_gate_consecutive_checks=10" in output
        else:
            assert "scheduler_cls=" not in output
            assert "sd_tail_gate_mode=" not in output

    roofline_output = _dry_run("qwen32b", "efficient_roofline_v2_k5")
    assert "sd_tail_gate_mode=roofline" in roofline_output
    assert "sd_tail_gate_margin=0.05" in roofline_output
    assert (
        "sd_tail_gate_config_path=/lustre/test/calibrations/qwen3.json"
        in roofline_output
    )


def test_matched_recipe_geometry_and_provenance_are_explicit() -> None:
    output = _dry_run("qwen32b", "always_on_v2_k5")

    for expected in (
        "grpo.max_num_steps=20",
        "checkpointing.enabled=false",
        "grpo.num_prompts_per_step=64",
        "grpo.num_generations_per_prompt=32",
        "policy.train_global_batch_size=512",
        "policy.max_total_sequence_length=4096",
        "policy.generation.max_new_tokens=4096",
        "policy.generation.vllm_cfg.max_model_len=4128",
        "max_num_batched_tokens=16384",
        "max_num_seqs=1024",
        "tensor_parallel_size=2",
        "draft_tensor_parallel_size=1",
        "moe_backend=triton",
        "WANDB_RUN_GROUP=contract",
        "logger.wandb.project=nemorl-vllm024-tail-gated-lyris",
        "logger.wandb.name=contract-attempt-1-qwen32b-always_on_v2_k5",
        "BASE_LOG_DIR=/lustre/test/tail-gate-runs/qwen32b/always_on_v2_k5",
    ):
        assert expected in output


def test_test_only_uses_lyris_scheduler_without_gres(tmp_path: Path) -> None:
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

    result = _run_launcher(
        "test-only",
        "qwen32b",
        "baseline_v2",
        REPO_DIR=str(REPO_ROOT),
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    assert "sbatch-args --test-only" in result.stdout
    assert "--segment=4" in result.stdout
    assert "--gres" not in result.stdout


def _create_pushed_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=sna/tail-gate", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in (
        [
            "git",
            "-C",
            str(repo),
            "config",
            "user.email",
            "launcher-test@example.invalid",
        ],
        ["git", "-C", str(repo), "config", "user.name", "Launcher Test"],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)
    for path in (
        repo / "ray.sub",
        repo / "experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh",
        repo / "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        repo / "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    subprocess.run(
        ["git", "-C", str(repo), "add", "."], check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "test: seed launcher repo"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "push", "--set-upstream", "origin", "sna/tail-gate"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def test_submit_records_complete_manifest_and_does_not_push(tmp_path: Path) -> None:
    repo, commit = _create_pushed_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nprintf '4242\\n'\n", encoding="utf-8")
    sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.write_text("container contract\n", encoding="utf-8")
    roofline = tmp_path / "roofline.json"
    roofline.write_text("{}\n", encoding="utf-8")
    draft_model = tmp_path / "draft-model"
    draft_model.mkdir()
    experiment_root = tmp_path / "runs"

    result = _run_launcher(
        "submit",
        "qwen32b",
        "efficient_roofline_v2_k5",
        REPO_DIR=str(repo),
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER=str(container),
        QWEN32_DRAFT_MODEL=str(draft_model),
        ROOFLINE_CONFIG=str(roofline),
        EXPERIMENT_ROOT=str(experiment_root),
        RUN_TAG="submit-contract",
        ATTEMPT_ID="attempt-1",
        WANDB_API_KEY="test-only-key",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    manifest = (experiment_root / "submissions.tsv").read_text(encoding="utf-8")
    assert {
        "runner",
        "graph_mode",
        "gate_mode",
        "k",
        "threshold",
        "consecutive_checks",
        "roofline_config_sha256",
        "commit",
        "container",
        "recipe",
        "job_id",
    }.issubset(manifest.splitlines()[0].split("\t"))
    assert "\tv2\tFULL_AND_PIECEWISE\troofline\t5\t32\t10\t" in manifest
    assert hashlib.sha256(roofline.read_bytes()).hexdigest() in manifest
    assert commit in manifest
    assert "4242" in manifest
    assert "test-only-key" not in manifest
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_submit_ignores_known_unit_artifacts_but_rejects_other_untracked_files(
    tmp_path: Path,
) -> None:
    repo, _commit = _create_pushed_repo(tmp_path)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    (repo / "tests/unit/unit_results").mkdir(parents=True)
    (repo / "tests/unit/unit_results.json").write_text("{}\n", encoding="utf-8")
    (repo / "tests/unit/unit_results/result.json").write_text("{}\n", encoding="utf-8")

    result = _run_launcher(
        "submit",
        "qwen30ba3b",
        "baseline_v1",
        REPO_DIR=str(repo),
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
    )
    assert result.returncode != 2 or "clean" not in result.stderr

    (repo / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")
    rejected = _run_launcher(
        "submit",
        "qwen30ba3b",
        "baseline_v1",
        REPO_DIR=str(repo),
        LYRIS_ROOT="/lustre/test",
        HF_HOME="/lustre/test/hf_home",
        CONTAINER=str(container),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        WANDB_API_KEY="test-only-key",
    )
    assert rejected.returncode == 2
    assert "untracked" in rejected.stderr
