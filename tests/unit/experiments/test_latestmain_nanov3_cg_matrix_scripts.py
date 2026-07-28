# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contracts for the explicit NanoV3 CUDA Graph Slurm submission scripts."""

import ast
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
SCRIPT_ROOT = REPO_ROOT / "experiments/cuda_graph/latestmain_nanov3"
SCOPE_ROOT = SCRIPT_ROOT / "scopes"
SETUP_TEST = REPO_ROOT / "tests/unit/models/megatron/test_megatron_setup.py"
RAY_SUB = REPO_ROOT / "ray.sub"
FULL_RUNTIME_PROBE = (
    REPO_ROOT / "experiments/cuda_graph/probe_latestmain_full_runtime.sbatch"
)
UV_OVERLAY_PROBE = (
    REPO_ROOT / "experiments/cuda_graph/probe_latestmain_uv_overlay.sbatch"
)


def _task3_valid_scope_cases() -> list[tuple[str, ...]]:
    """Read Task 3's matrix without importing GPU-dependent test modules."""
    tree = ast.parse(SETUP_TEST.read_text())
    valid_scope_cases = next(
        assignment.value
        for assignment in tree.body
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "VALID_SCOPE_CASES"
            for target in assignment.targets
        )
    )
    return ast.literal_eval(valid_scope_cases)


VALID_SCOPE_CASES = _task3_valid_scope_cases()
SCOPE_SCRIPTS = {
    "00_nocg.sh": (),
    "01_attn.sh": ("attn",),
    "02_mamba.sh": ("mamba",),
    "03_attn_mamba.sh": ("attn", "mamba"),
    "04_moe.sh": ("moe",),
    "05_attn_moe.sh": ("attn", "moe"),
    "06_mamba_moe.sh": ("mamba", "moe"),
    "07_attn_mamba_moe.sh": ("attn", "mamba", "moe"),
    "08_moe_router.sh": ("moe_router",),
    "09_attn_moe_router.sh": ("attn", "moe_router"),
    "10_mamba_moe_router.sh": ("mamba", "moe_router"),
    "11_attn_mamba_moe_router.sh": ("attn", "mamba", "moe_router"),
    "12_moe_router_preprocess.sh": ("moe_router", "moe_preprocess"),
    "13_attn_moe_router_preprocess.sh": (
        "attn",
        "moe_router",
        "moe_preprocess",
    ),
    "14_mamba_moe_router_preprocess.sh": (
        "mamba",
        "moe_router",
        "moe_preprocess",
    ),
    "15_attn_mamba_moe_router_preprocess.sh": (
        "attn",
        "mamba",
        "moe_router",
        "moe_preprocess",
    ),
}


def _script_source(script_name: str) -> str:
    return (SCOPE_ROOT / script_name).read_text()


def _visible_scope(script_source: str) -> tuple[str, ...]:
    match = re.search(r"cuda_graph_scope=\[([^]]*)\]", script_source)
    if match is None:
        return ()
    return tuple(item for item in match.group(1).split(",") if item)


def _fake_sbatch(tmp_path: Path) -> Path:
    sbatch_path = tmp_path / "sbatch"
    sbatch_path.write_text(
        "#!/usr/bin/env bash\nprintf 'fake-sbatch %q ' \"$@\"\nprintf '\\n'\n"
    )
    sbatch_path.chmod(0o755)
    return sbatch_path


def _run_test_only(
    script_name: str, tmp_path: Path, **extra_env: str
) -> subprocess.CompletedProcess[str]:
    _fake_sbatch(tmp_path)
    env = (
        os.environ
        | {
            "CLUSTER": "ptyche",
            "PHASE": "smoke",
            "TEST_ONLY": "1",
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
        }
        | extra_env
    )
    return subprocess.run(
        ["bash", str(SCOPE_ROOT / script_name)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def test_scope_scripts_are_exact_task3_matrix_with_visible_scope_per_file() -> None:
    """Every reusable file has the sole, hard-coded scope it submits."""
    assert list(SCOPE_SCRIPTS.values()) == VALID_SCOPE_CASES
    assert sorted(path.name for path in SCOPE_ROOT.glob("*.sh")) == sorted(
        SCOPE_SCRIPTS
    )

    for script_name, expected_scope in SCOPE_SCRIPTS.items():
        source = _script_source(script_name)
        assert "CONFIG=examples/configs/recipes/llm/performance/" in source
        assert "RUN_NAME=" in source
        assert _visible_scope(source) == expected_scope


def test_non_cg_scripts_keep_required_te_graph_settings_visible() -> None:
    """Graph conditions make their TE packed-THD settings easy to diff."""
    for script_name in list(SCOPE_SCRIPTS)[1:]:
        source = _script_source(script_name)
        assert "cuda_graph_impl=transformer_engine" in source
        assert "cuda_graph_packed_seq=true" in source
        assert "cuda_graph_warmup_steps=3" in source
        assert "checkpointing.enabled=false" in source
        assert "sbatch" in source
        assert "ray.sub" in source


def test_nocg_script_emits_no_training_cuda_graph_override() -> None:
    """The baseline remains eager even when the shared recipe changes later."""
    source = _script_source("00_nocg.sh")
    assert "cuda_graph_" not in source
    assert "checkpointing.enabled=false" in source
    assert "ray.sub" in source


def test_shared_ray_submission_template_has_no_job_dependency() -> None:
    """Independent scope submissions must not be serialized by ray.sub."""
    assert "--dependency" not in RAY_SUB.read_text()


def test_all_scope_dry_run_prints_test_only_submission_with_exact_scope(
    tmp_path: Path,
) -> None:
    """The longest valid condition preserves its literal four-module scope."""
    result = _run_test_only("15_attn_mamba_moe_router_preprocess.sh", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "cuda_graph_scope=[attn,mamba,moe_router,moe_preprocess]" in result.stdout
    assert "sbatch --test-only" in result.stdout
    assert "fake-sbatch --test-only" in result.stdout
    assert "--gpus-per-node" not in result.stdout
    assert "--gres=gpu" not in result.stdout


def test_moe_scope_scripts_validate_explicit_binary_experiment_axes(
    tmp_path: Path,
) -> None:
    """MoE extensions vary only overlap/recompute, never the hard-coded scope."""
    valid = _run_test_only(
        "08_moe_router.sh",
        tmp_path,
        SHARED_EXPERT_OVERLAP="1",
        MOE_ACT_RECOMPUTE="1",
    )
    assert valid.returncode == 0, valid.stderr
    assert "moe_shared_expert_overlap=true" in valid.stdout
    assert "recompute_modules=[moe_act]" in valid.stdout
    assert "-shared-expert-overlap-moe-act-recompute" in valid.stdout
    assert "cuda_graph_scope=[moe_router]" in valid.stdout

    invalid = _run_test_only("08_moe_router.sh", tmp_path, SHARED_EXPERT_OVERLAP="yes")
    assert invalid.returncode == 2
    assert "SHARED_EXPERT_OVERLAP must be 0 or 1" in invalid.stderr


def test_profiles_have_stable_container_paths_and_live_cluster_gpu_requests() -> None:
    """Ptyche relies on its whole-node allocation while OCI retains its GRES request."""
    ptyche = (SCRIPT_ROOT / "profiles/ptyche.env").read_text()
    oci_hsg = (SCRIPT_ROOT / "profiles/oci-hsg.env").read_text()
    assert re.search(r"^CONTAINER=\S+", ptyche, re.MULTILINE)
    assert re.search(r"^CONTAINER=\S+", oci_hsg, re.MULTILINE)
    assert "SBATCH_GPU_ARGS=()" in ptyche
    assert not re.search(
        r"^SBATCH_GPU_ARGS=\([^)]*--gpus-per-node", ptyche, re.MULTILINE
    )
    assert "SBATCH_GPU_ARGS=(--gres" not in ptyche
    assert "SBATCH_GPU_ARGS=(--gres=gpu:4)" in oci_hsg


def test_ptyche_profile_uses_the_staged_nightly_stable_link() -> None:
    """Ptyche model jobs consume the immutable-nightly stable link, never July's image."""
    ptyche = (SCRIPT_ROOT / "profiles/ptyche.env").read_text()
    assert (
        "CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly.sqsh"
        in ptyche
    )
    assert "nemo_rl_latestmain_nanov3.sqsh" not in ptyche


def test_full_runtime_probe_checks_the_required_gpu_and_package_gates() -> None:
    """The pre-matrix probe must reject incomplete nightly images before model jobs."""
    source = FULL_RUNTIME_PROBE.read_text()
    assert "#SBATCH --exclusive" in source
    assert "torch.cuda.is_available()" in source
    assert "sys.version_info >= (3, 13, 13)" in source
    assert "uv lock --check" in source
    assert "transformer_engine.pytorch" in source
    assert "megatron.core" in source
    assert "mamba_ssm" in source
    assert "nemo_rl" in source
    assert "load_config" in source
    assert "cuda_graph_warmup_steps=3" in source
    assert "#SBATCH --gres" not in source


def test_uv_overlay_probe_preserves_the_lock_and_tests_the_actual_launcher_path() -> (
    None
):
    """The recovery probe records lock drift without rewriting source state."""
    source = UV_OVERLAY_PROBE.read_text()
    assert "#SBATCH --exclusive" in source
    assert "#SBATCH --gres" not in source
    assert "uv lock --check" in source
    assert "NRL_FORCE_REBUILD_VENVS=true uv run --frozen python -c" in source
    assert "transformer_engine.pytorch" in source
    assert "megatron.core" in source
    assert "nemo_rl" in source


def test_submit_all_invokes_each_named_file_without_dynamic_scope_rewrite() -> None:
    """The matrix launcher selects committed scripts instead of constructing scopes."""
    source = (SCRIPT_ROOT / "submit_all_valid_scopes.sh").read_text()
    assert "cuda_graph_scope" not in source
    for script_name in SCOPE_SCRIPTS:
        assert script_name in source
