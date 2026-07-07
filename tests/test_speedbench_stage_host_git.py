from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = (
    REPO_ROOT / "experiments" / "vllm_024_dynamicsd" / "stage_speedbench.sh"
)


def test_modelopt_git_staging_runs_outside_container_payload() -> None:
    text = STAGE_SCRIPT.read_text(encoding="utf-8")
    payload_start = text.index("read -r -d '' PAYLOAD")
    payload_end = text.index('srun "${srun_args[@]}"')
    host_setup = text[:payload_start]
    container_payload = text[payload_start:payload_end]

    assert 'git clone --filter=blob:none "$MODELOPT_REPO_URL"' in host_setup
    assert 'git -C "$MODELOPT_WORK_ROOT" checkout "$MODELOPT_REVISION"' in host_setup
    assert "git clone" not in container_payload
    assert 'git -C "$MODELOPT_WORK_ROOT"' not in container_payload
    assert 'MODELOPT_SOURCE_ROOT="$SOURCE_ROOT/modelopt"' in host_setup
    assert 'python3 "$MODELOPT_SOURCE_ROOT/$MODELOPT_PREPARE_DATA_SCRIPT"' in container_payload
