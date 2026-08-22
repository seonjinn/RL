import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_CALL_LOG": str(tmp_path / "sbatch.log"),
        "REMOTE_REPO": "/home/sna/nemo-rl-pack-cp2-baseline",
        "EXPECTED_HEAD": "a" * 40,
        "FINAL_ROOT": "/lustre/fake/pack-cp2-baseline",
        "CONTAINER": "/lustre/fake/container.sqsh",
        "TARGET_SNAPSHOT": "/lustre/target/b968826d9c46dd6066d109eabc6255188de91218",
        "DRAFTER_SNAPSHOT": "/lustre/draft/9b41424b7109f9c5413454f481b09a82b85333f4",
        "SBATCH_ACCOUNT": "nemotron_n3_post",
        "WANDB_API_KEY": "test-only-placeholder",  # pragma: allowlist secret
    }


def test_test_only_forecasts_exactly_three_rotated_pairs(tmp_path: Path) -> None:
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALL_LOG"\n'
        'case " $* " in *" --test-only "*) echo forecast-ok >&2 ;; '
        '*) exit 91 ;; esac\n'
    )
    sbatch.chmod(0o755)

    result = subprocess.run(
        ["bash", ROOT / "submit_matrix.sh", "--test-only"],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(tmp_path),
    )
    calls = (tmp_path / "sbatch.log").read_text().splitlines()

    assert len(calls) == 3
    assert all("--test-only" in call for call in calls)
    assert all("--partition=batch" in call for call in calls)
    assert all("--partition=batch_long" not in call for call in calls)
    assert all("--time=04:00:00" in call for call in calls)
    assert "REPLICATE=1" in calls[0] and "FIRST_ARM=fixed" in calls[0]
    assert "REPLICATE=2" in calls[1] and "FIRST_ARM=online" in calls[1]
    assert "REPLICATE=3" in calls[2] and "FIRST_ARM=fixed" in calls[2]
    assert len({token for call in calls for token in call.split(",") if "WANDB_RUN_ID=" in token}) == 6
    assert "jobs_submitted=0" in result.stdout


def test_monitor_rejects_duplicate_job_ids(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", ROOT / "monitor_matrix.sh", "101", "101", "102"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "duplicate job ID" in result.stderr
