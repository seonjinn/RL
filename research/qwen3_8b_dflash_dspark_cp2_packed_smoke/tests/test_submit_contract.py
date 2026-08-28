import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_CALL_LOG": str(tmp_path / "sbatch.log"),
        "REMOTE_REPO": "/home/sna/nemo-rl-q8-cp2-smoke",
        "EXPECTED_HEAD": "a" * 40,
        "FINAL_ROOT": "/lustre/fake/q8-cp2-smoke",
        "CONTAINER": "/lustre/fake/nemo-rl.sqsh",
        "CONTAINER_SHA256": "b" * 64,
        "TARGET_SNAPSHOT": "/lustre/target/b968826d9c46dd6066d109eabc6255188de91218",
        "DFLASH_SNAPSHOT": "/lustre/dflash/9b41424b7109f9c5413454f481b09a82b85333f4",
        "DSPARK_SNAPSHOT": "/lustre/dspark/03326e5043815da1f81b109078b2889737c26017",
        "SBATCH_ACCOUNT": "nemotron_n3_post",
        "WANDB_API_KEY": "test-only-placeholder",  # pragma: allowlist secret
        "WANDB_PROJECT": "sna-specdec-cp2-validation",
    }


def test_test_only_forecasts_exactly_two_provider_jobs(tmp_path: Path) -> None:
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALL_LOG"\n'
        'case " $* " in *" --test-only "*) echo forecast-ok >&2 ;; '
        "*) exit 91 ;; esac\n"
    )
    sbatch.chmod(0o755)

    result = subprocess.run(
        ["bash", ROOT / "submit_oci_hsg.sh", "--test-only"],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(tmp_path),
    )
    calls = (tmp_path / "sbatch.log").read_text().splitlines()

    assert len(calls) == 2
    assert all("--test-only" in call for call in calls)
    assert all("--account=nemotron_n3_post" in call for call in calls)
    assert all("--partition=batch" in call for call in calls)
    assert all("--qos=normal" in call for call in calls)
    assert all("--nodes=1" in call and "--gres=gpu:4" in call for call in calls)
    assert "ARM=dflash" in calls[0]
    assert "ARM=dspark" in calls[1]
    assert "jobs_submitted=0" in result.stdout


def test_submit_mode_rejects_duplicate_job_ids(tmp_path: Path) -> None:
    sbatch = tmp_path / "sbatch"
    sbatch.write_text("#!/bin/sh\necho '7001;cluster'\n")
    sbatch.chmod(0o755)

    result = subprocess.run(
        ["bash", ROOT / "submit_oci_hsg.sh"],
        capture_output=True,
        text=True,
        env=_environment(tmp_path),
    )

    assert result.returncode != 0
    assert "duplicate job IDs" in result.stderr
