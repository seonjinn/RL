from pathlib import Path


GATE = Path(__file__).with_name("task6_linux_gate.sbatch")
EXPECTED_HEAD = "ce83f01ea54641435f9ab17f09caf087195485e6"
SOURCE_DIR = "/home/sna/nemo-rl-task6-ce83f01e"
RESULT_ROOT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "cadence-task6-ce83f01e"
)


def _validate(text: str) -> None:
    combined = "uv sync --locked --extra mcore --extra vllm --group test --group dev"
    assert combined not in text, "RED: mutually conflicting extras are combined"
    assert text.count("uv sync --locked --extra mcore --group test --group dev") == 1
    assert text.count("uv sync --locked --extra vllm --group test --group dev") == 1
    assert "mcore-venv" in text and "vllm-venv" in text
    assert f"expected_head={EXPECTED_HEAD}" in text
    assert f"source_dir={SOURCE_DIR}" in text
    assert f"result_root={RESULT_ROOT}" in text
    assert "#SBATCH --account=nemotron_n3_post" in text
    assert "/tmp/nr${SLURM_JOB_ID}-m" in text
    assert "/tmp/nr${SLURM_JOB_ID}-v" in text
    assert text.count("unset RAY_ADDRESS") >= 2
    assert text.count('test "${#RAY_TMPDIR}" -le 32') >= 2
    assert text.count('rm -rf -- "${RAY_TMPDIR}"') >= 2
    assert "task6_linux_test_compat_contract.py" in text
    assert "import megatron.core" in text
    assert "import vllm" in text
    assert "phase-mcore-${SLURM_JOB_ID}.txt" in text
    assert "phase-vllm-${SLURM_JOB_ID}.txt" in text
    assert text.count("phase_exit_code=%s") >= 2
    assert "TASK6_MCORE_PHASE_PASS" in text
    assert "TASK6_VLLM_PHASE_PASS" in text
    assert "TASK6_LINUX_GATE_PASS" in text


def main() -> None:
    text = GATE.read_text()
    _validate(text)

    negative_cases = {
        "combined extras": text.replace(
            "uv sync --locked --extra mcore --group test --group dev",
            "uv sync --locked --extra mcore --extra vllm --group test --group dev",
            1,
        ),
        "long Ray path": text.replace(
            "/tmp/nr${SLURM_JOB_ID}-m",
            "/raid/scratch/sna/cadence-task6-ce83f01e-${SLURM_JOB_ID}/ray-mcore",
        ),
        "stale product head": text.replace(EXPECTED_HEAD, "bbea05d3" * 5, 1),
    }
    for name, invalid_text in negative_cases.items():
        try:
            _validate(invalid_text)
        except AssertionError:
            continue
        raise AssertionError(f"RED: dynamic negative case passed: {name}")
    print("TASK6_SPLIT_GATE_CONTRACT_GREEN")


if __name__ == "__main__":
    main()
