import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_refit_perf_matrix"


def test_runner_pins_correctness_verified_product_head() -> None:
    runner = (EXPERIMENT_DIR / "run_pair_oci_hsg.sbatch").read_text()

    assert "readonly product_head=0f712654329acdb3693dd53c1453b49c6b9c1ce9" in runner


def test_runner_expands_config_path_in_container_shell() -> None:
    runner = (EXPERIMENT_DIR / "run_pair_oci_hsg.sbatch").read_text()

    assert '--config \\"${REMOTE_REPO}/\\${config_path}\\"' in runner
    assert "--config '${REMOTE_REPO}/\\${config_path}'" not in runner


def test_runner_ray_tmpdir_fits_unix_socket_limit() -> None:
    runner = (EXPERIMENT_DIR / "run_pair_oci_hsg.sbatch").read_text()
    match = re.search(r'readonly ray_root="([^"]+)"', runner)
    assert match is not None
    ray_root = match.group(1).replace("${SLURM_JOB_ID}", "6432875")
    assert ray_root.startswith("/raid/scratch/")
    socket_path = (
        f"{ray_root}/online/ray/session_2026-08-21_18-50-33_347689_1380691/"
        "sockets/plasma_store"
    )

    assert len(os.fsencode(socket_path)) <= 107


def test_test_only_forecasts_nine_same_node_pairs_without_submitting(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "sbatch.log"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALL_LOG"\n'
        'case " $* " in *" --test-only "*) echo "forecast ok" >&2 ;; '
        '*) echo "unexpected actual submission" >&2; exit 91 ;; esac\n'
    )
    sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_CALL_LOG": str(calls),
        "REMOTE_REPO": "/home/sna/nemo-rl-matrix",
        "EXPECTED_HEAD": "a" * 40,
        "FINAL_ROOT": "/lustre/fake/matrix",
        "CONTAINER": "/lustre/fake/container.sqsh",
        "TARGET_SNAPSHOT": "/lustre/fake/target/b968826d9c46dd6066d109eabc6255188de91218",
        "DRAFTER_SNAPSHOT": "/lustre/fake/draft/9b41424b7109f9c5413454f481b09a82b85333f4",
        "SBATCH_ACCOUNT": "test-account",
        "WANDB_API_KEY": "test-only-placeholder",  # pragma: allowlist secret
    }

    result = subprocess.run(
        ["bash", EXPERIMENT_DIR / "submit_matrix.sh", "--test-only"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    submitted = calls.read_text().splitlines()
    assert len(submitted) == 9
    assert all("--test-only" in call for call in submitted)
    assert "PAIR_SHAPE=gbs32_mbs1" in submitted[0]
    assert "FIRST_ARM=fixed" in submitted[0]
    assert "REPLICATE=1" in submitted[0]
    assert "PAIR_SHAPE=gbs32_mbs1" in submitted[1]
    assert "FIRST_ARM=online" in submitted[1]
    assert "REPLICATE=2" in submitted[1]
    assert "PAIR_SHAPE=gbs32_mbs1" in submitted[2]
    assert "FIRST_ARM=fixed" in submitted[2]
    assert "REPLICATE=3" in submitted[2]
    assert "PAIR_SHAPE=gbs64_mbs1" in submitted[3]
    assert "PAIR_SHAPE=gbs64_mbs2" in submitted[6]
    assert all("FIXED_WANDB_RUN_ID=" in call for call in submitted)
    assert all("ONLINE_WANDB_RUN_ID=" in call for call in submitted)
    final_dirs = [
        token
        for call in submitted
        for token in call.split(",")
        if token.startswith("FINAL_DIR=")
    ]
    assert len(set(final_dirs)) == 9
    fixed_ids = [
        token
        for call in submitted
        for token in call.split(",")
        if token.startswith("FIXED_WANDB_RUN_ID=")
    ]
    online_ids = [
        token
        for call in submitted
        for token in call.split(",")
        if token.startswith("ONLINE_WANDB_RUN_ID=")
    ]
    assert len(set(fixed_ids + online_ids)) == 18
    assert all(
        f"FIXED_WANDB_RUN_ID=q8-{shape}-r{replicate}-fixed-" in submitted[index]
        and f"ONLINE_WANDB_RUN_ID=q8-{shape}-r{replicate}-online-" in submitted[index]
        for index, (shape, replicate) in enumerate(
            (shape, replicate)
            for shape in ("gbs32_mbs1", "gbs64_mbs1", "gbs64_mbs2")
            for replicate in (1, 2, 3)
        )
    )
    assert "submission_mode=test-only jobs_submitted=0" in result.stdout


def test_monitor_observes_all_jobs_for_five_sixty_second_passes(
    tmp_path: Path,
) -> None:
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    sacct = tmp_path / "sacct"
    sacct.write_text(
        "#!/bin/sh\n"
        'printf "101|pair-a|RUNNING|0:0|00:01:00\\n"\n'
        'printf "102|pair-b|PENDING|0:0|00:00:00\\n"\n'
        'printf "103|pair-c|RUNNING|0:0|00:01:00\\n"\n'
    )
    sacct.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", EXPERIMENT_DIR / "monitor_matrix.sh", "101", "102", "103"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.count("monitoring_pass=") == 5
    assert "five_minute_monitor_complete=yes" in result.stdout
