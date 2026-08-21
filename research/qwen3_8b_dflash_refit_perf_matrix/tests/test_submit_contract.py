import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_refit_perf_matrix"


def test_test_only_forecasts_three_same_node_pairs_without_submitting(
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
    assert len(submitted) == 3
    assert all("--test-only" in call for call in submitted)
    assert "PAIR_SHAPE=gbs32_mbs1" in submitted[0]
    assert "FIRST_ARM=fixed" in submitted[0]
    assert "PAIR_SHAPE=gbs64_mbs1" in submitted[1]
    assert "FIRST_ARM=online" in submitted[1]
    assert "PAIR_SHAPE=gbs64_mbs2" in submitted[2]
    assert "FIRST_ARM=fixed" in submitted[2]
    assert all("FIXED_WANDB_RUN_ID=" in call for call in submitted)
    assert all("ONLINE_WANDB_RUN_ID=" in call for call in submitted)
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
