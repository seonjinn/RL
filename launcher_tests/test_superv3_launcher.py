from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "launch_superv3_main_short.sh"


def test_superv3_launcher_forwards_an_explicit_slurm_nodelist() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'if [[ -n "${SLURM_NODELIST:-}" ]]; then' in launcher
    assert 'SBATCH_ARGS+=(--nodelist="${SLURM_NODELIST}")' in launcher
