from pathlib import Path


EXPERIMENT_DIR = (
    Path(__file__).parents[3]
    / "experiments"
    / "cuda_graph"
    / "mamba_moe_te_graph_20260729"
)
SCRIPT_PATH = EXPERIMENT_DIR / "scripts" / "build_te_pr2898_backport.sub"
TE_SOURCE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/"
    "TransformerEngine-fp64-thd-cudagraph-20260730"
)
TE_COMMIT = "ba256c5b23c8f19b64a0c26499277d15c4133a1c"
IMAGE = "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260729_2472184.sqsh"
IMAGE_SHA256 = "cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91"


def test_te_pr2898_build_launcher_pins_host_and_container_provenance() -> None:
    script = SCRIPT_PATH.read_text()

    assert f"TE_SOURCE={TE_SOURCE}" in script
    assert f"EXPECTED_TE_COMMIT={TE_COMMIT}" in script
    assert f"IMAGE={IMAGE}" in script
    assert f"IMAGE_SHA256={IMAGE_SHA256}" in script
    assert (
        'test "$(sha256sum -- "${IMAGE}" | awk \'{print $1}\')" = "${IMAGE_SHA256}"'
        in script
    )
    assert (
        'test "$(git -C "${TE_SOURCE}" rev-parse HEAD)" = "${EXPECTED_TE_COMMIT}"'
        in script
    )
    assert 'git -C "${TE_SOURCE}" diff --quiet' in script
    assert 'git -C "${TE_SOURCE}" diff --cached --quiet' in script
    assert 'git -C "${TE_SOURCE}" status --porcelain --untracked-files=all' in script
    assert 'git -C "${TE_SOURCE}" submodule status --recursive' in script
    assert script.index("sha256sum --") < script.index("srun --nodes=1 --ntasks=1")
    assert script.index("rev-parse HEAD") < script.index("srun --nodes=1 --ntasks=1")


def test_te_pr2898_build_launcher_is_offline_bounded_and_atomically_publishes_one_wheel() -> (
    None
):
    script = SCRIPT_PATH.read_text()

    assert "NVTE_FRAMEWORK=pytorch" in script
    assert "NVTE_CUDA_ARCHS=100" in script
    assert "TE_BUILD_JOBS=${TE_BUILD_JOBS:-8}" in script
    assert "TE_BUILD_JOBS must be an integer from 1 through 16" in script
    assert "MAX_JOBS=${TE_BUILD_JOBS}" in script
    assert "PIP_NO_INDEX=1" in script
    assert "PIP_NO_BUILD_ISOLATION=1" in script
    assert "/opt/nemo_rl_venv/bin/python setup.py bdist_wheel --dist-dir" in script
    assert "pip install" not in script
    assert "uv sync" not in script
    assert "--checkpoint" not in script
    assert "mapfile -t wheels" in script
    assert "Expected exactly one wheel" in script
    assert "wheel_sha256" in script
    assert "provenance.json" in script
    assert 'mv -- "${STAGING_DIR}" "${OUTPUT_DIR}"' in script
    assert "transformer-engine-pr2898-${EXPECTED_TE_COMMIT}" in script


def test_te_pr2898_build_stays_out_of_performance_launchers_and_is_documented() -> None:
    runner = (EXPERIMENT_DIR / "run_scope.sh").read_text()
    performance_submitter = (EXPERIMENT_DIR / "submit_performance.sh").read_text()
    readme = (EXPERIMENT_DIR / "README.md").read_text()

    for text in (runner, performance_submitter):
        assert "setup.py bdist_wheel" not in text
        assert "build_te_pr2898_backport" not in text
    assert "build_te_pr2898_backport.sub" in readme
    assert TE_COMMIT in readme
