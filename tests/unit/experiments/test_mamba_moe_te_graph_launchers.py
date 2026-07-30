import hashlib
import importlib.util
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

EXPERIMENT_DIR = (
    Path(__file__).parents[3]
    / "experiments"
    / "cuda_graph"
    / "mamba_moe_te_graph_20260729"
)
DENSE_AXES = ("attn", "mlp", "mamba")
MOE_AXES = (
    (),
    ("moe",),
    ("moe_router",),
    ("moe_router", "moe_preprocess"),
)
VALID_GRAPH_SCOPES = {
    tuple(
        name for enabled, name in zip(enabled_dense, DENSE_AXES, strict=True) if enabled
    )
    + moe_scope
    for enabled_dense in itertools.product((False, True), repeat=3)
    for moe_scope in MOE_AXES
}
TE_FP64_WEAKREF_COMMIT = "e707aa46869dc2aec08dfea25402e97a61d49fef"
TE_FP64_WEAKREF_SHA256 = (
    "39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b"
)
TE_FP64_WEAKREF_SOURCE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/"
    "TransformerEngine-fp64-weakref-20260729/.overlay/"
    f"{TE_FP64_WEAKREF_COMMIT}/utils.py"
)
TE_FP64_WEAKREF_TARGET = (
    "/root/.cache/uv/archive-v0/AdbVCNRp6JVFPo0e/"
    "transformer_engine/pytorch/utils.py"
)
TE_EXPECTED_VERSION = "2.15.0+42b84005"


def _assignment(script: Path, name: str) -> str:
    match = re.search(
        rf"(?:^|[ \n]){name}=(?:'([^']*)'|([^ \n]+))",
        script.read_text(),
        re.MULTILINE,
    )
    assert match is not None, f"{script.name} does not set {name}"
    return match.group(1) or match.group(2)


def _scope(script: Path) -> tuple[str, ...]:
    value = _assignment(script, "SCOPE")
    assert value.startswith("[") and value.endswith("]")
    return tuple(part for part in value[1:-1].split(",") if part)


def _run_script(
    relative_path: str,
    **environment: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        ["bash", str(EXPERIMENT_DIR / relative_path)],
        cwd=EXPERIMENT_DIR.parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _load_experiment_module(name: str) -> ModuleType:
    path = EXPERIMENT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXPERIMENT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_scope_matrix_is_complete_and_exact() -> None:
    assert len(VALID_GRAPH_SCOPES) == 32
    assert EXPERIMENT_DIR.is_dir()

    scripts = sorted((EXPERIMENT_DIR / "scopes").glob("*.sh"))
    baseline = [script for script in scripts if script.name == "00_baseline_no_cg.sh"]
    te_scripts = [script for script in scripts if script not in baseline]

    assert len(baseline) == 1
    assert len(te_scripts) == 32
    assert {_scope(script) for script in te_scripts} == VALID_GRAPH_SCOPES
    assert _assignment(baseline[0], "CUDA_GRAPH_IMPL") == "none"
    assert _assignment(baseline[0], "SCOPE") == "[no_cg]"


def test_scope_scripts_pin_graph_and_run_contracts() -> None:
    scripts = sorted((EXPERIMENT_DIR / "scopes").glob("*.sh"))
    run_names = [_assignment(script, "RUN_NAME") for script in scripts]

    assert len(run_names) == len(set(run_names)) == 33
    for script in scripts:
        assert _assignment(script, "WARMUP_STEPS") == "3"
        assert _assignment(script, "CACHE_CAPACITY") == "2"
        assert _assignment(script, "MAX_PACKED_SEQS") == "16"
        assert _assignment(script, "CHECKPOINTING_ENABLED") == "false"
        assert _assignment(script, "WANDB_PROJECT") == "sna-cg-study"
        assert (
            'bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"'
            in script.read_text()
        )

    for script in scripts[1:]:
        assert _assignment(script, "CUDA_GRAPH_IMPL") == "transformer_engine"


def test_moe_configuration_variants_are_persistent_and_not_graph_scopes() -> None:
    variants = sorted((EXPERIMENT_DIR / "variants").glob("*.sh"))
    expected = {
        (scope, overlap, moe_act)
        for scope in (
            ("moe",),
            ("moe_router", "moe_preprocess"),
        )
        for overlap in ("false", "true")
        for moe_act in ("false", "true")
    }
    actual = {
        (
            _scope(script),
            _assignment(script, "MOE_SHARED_EXPERT_OVERLAP"),
            _assignment(script, "MOE_ACT_RECOMPUTE"),
        )
        for script in variants
    }

    assert len(variants) == 8
    assert actual == expected
    assert len({_assignment(script, "RUN_NAME") for script in variants}) == 8
    for script in variants:
        assert "moe_act" not in _scope(script)
        assert "shared_expert" not in _scope(script)
        assert _assignment(script, "CHECKPOINTING_ENABLED") == "false"


def test_test_only_reports_resolved_ptyche_nano_provenance_and_never_submits() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    assert "UNRESOLVED: none" in result.stdout
    assert "COMMAND:" in result.stdout
    assert "SBATCH:" in result.stdout
    assert (
        "examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml"
        in result.stdout
    )
    assert "+policy.megatron_cfg.cuda_graph_modules=\\[attn\\]" in result.stdout
    assert "+policy.megatron_cfg.cuda_graph_warmup_steps=3" in result.stdout
    assert "+policy.megatron_cfg.cuda_graph_max_cached_schedules=2" in result.stdout
    assert "+policy.megatron_cfg.cuda_graph_max_packed_seqs=16" in result.stdout
    assert "checkpointing.enabled=false" in result.stdout
    assert (
        "+checkpointing.pretrained_checkpoint.path="
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/checkpoints/"
        "nanov3-30b-a3b-pr5672-20260720/nvidia/"
        "NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16" in result.stdout
    )
    assert (
        "+checkpointing.pretrained_checkpoint.format=megatron_lm"
        in result.stdout
    )
    assert "logger.wandb.project=sna-cg-study" in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout


@pytest.mark.parametrize(
    "launcher",
    ["scopes/00_baseline_no_cg.sh", "scopes/17_attn.sh"],
)
def test_fp64_overlay_provenance_is_identical_for_baseline_and_te_scopes(
    launcher: str,
) -> None:
    result = _run_script(launcher, CLUSTER="ptyche", TEST_ONLY="1")

    assert result.returncode == 0, result.stderr
    for field, value in (
        ("TE_FP64_WEAKREF_COMMIT", TE_FP64_WEAKREF_COMMIT),
        ("TE_FP64_WEAKREF_SHA256", TE_FP64_WEAKREF_SHA256),
        ("TE_FP64_WEAKREF_SOURCE", TE_FP64_WEAKREF_SOURCE),
        ("TE_FP64_WEAKREF_TARGET", TE_FP64_WEAKREF_TARGET),
        ("TE_EXPECTED_VERSION", TE_EXPECTED_VERSION),
    ):
        assert f"{field}: {value}" in result.stdout
    assert f"{TE_FP64_WEAKREF_SOURCE}:{TE_FP64_WEAKREF_TARGET}:ro" in result.stdout
    assert "validate_te_fp64_overlay.py" in result.stdout
    assert f"--expected-version {TE_EXPECTED_VERSION}" in result.stdout
    assert f"--expected-sha256 {TE_FP64_WEAKREF_SHA256}" in result.stdout


def test_fp64_overlay_rejects_wrong_sha_before_cuda_preflight(
    tmp_path: Path,
) -> None:
    validator_path = EXPERIMENT_DIR / "validate_te_fp64_overlay.py"
    assert validator_path.is_file(), "FP64 overlay validator must be committed"

    package_root = tmp_path / "packages"
    transformer_engine_root = package_root / "transformer_engine"
    pytorch_root = transformer_engine_root / "pytorch"
    pytorch_root.mkdir(parents=True)
    (package_root / "torch.py").write_text(
        "float64 = object()\n"
        "def arange(*args, **kwargs):\n"
        "    raise RuntimeError('CUDA preflight ran')\n"
    )
    (transformer_engine_root / "__init__.py").write_text(
        f"__version__ = {TE_EXPECTED_VERSION!r}\n"
    )
    (pytorch_root / "__init__.py").touch()
    (pytorch_root / "utils.py").write_text(
        "import torch\n"
        "_torch_dtype_to_np_typestr_dict = {torch.float64: '<f8'}\n"
        "def make_weak_ref(tensor):\n"
        "    return tensor\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [
            sys.executable,
            str(validator_path),
            "--expected-version",
            TE_EXPECTED_VERSION,
            "--expected-sha256",
            "0" * 64,
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert "CUDA preflight ran" not in result.stderr


def test_fp64_overlay_rejects_writable_source_before_cuda_preflight(
    tmp_path: Path,
) -> None:
    validator_path = EXPERIMENT_DIR / "validate_te_fp64_overlay.py"
    package_root = tmp_path / "packages"
    transformer_engine_root = package_root / "transformer_engine"
    pytorch_root = transformer_engine_root / "pytorch"
    pytorch_root.mkdir(parents=True)
    (package_root / "torch.py").write_text(
        "float64 = object()\n"
        "def arange(*args, **kwargs):\n"
        "    raise RuntimeError('CUDA preflight ran')\n"
    )
    (transformer_engine_root / "__init__.py").write_text(
        f"__version__ = {TE_EXPECTED_VERSION!r}\n"
    )
    (pytorch_root / "__init__.py").touch()
    utils_path = pytorch_root / "utils.py"
    utils_path.write_text(
        "import torch\n"
        "_torch_dtype_to_np_typestr_dict = {torch.float64: '<f8'}\n"
        "def make_weak_ref(tensor):\n"
        "    return tensor\n"
    )
    utils_path.chmod(0o644)
    expected_sha256 = hashlib.sha256(utils_path.read_bytes()).hexdigest()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [
            sys.executable,
            str(validator_path),
            "--expected-version",
            TE_EXPECTED_VERSION,
            "--expected-sha256",
            expected_sha256,
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "mode mismatch" in result.stderr
    assert "CUDA preflight ran" not in result.stderr


@pytest.mark.parametrize("phase", ["profile", "benchmark"])
def test_runner_rejects_invalid_phases(phase: str) -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        PHASE=phase,
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert "PHASE must be smoke, performance, or accuracy" in result.stderr


def test_runner_rejects_unknown_cluster_profiles() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="unknown",
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert "CLUSTER must be ptyche or oci-hsg" in result.stderr


def test_runner_rejects_qwen_mamba_scopes_before_submission() -> None:
    result = _run_script(
        "scopes/05_mamba.sh",
        CLUSTER="ptyche",
        MODEL="qwen3-30b-a3b",
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert "has no Mamba layers" in result.stderr
    assert "SBATCH:" not in result.stdout


@pytest.mark.parametrize(
    ("model", "recipe"),
    [
        (
            "qwen3-30b-a3b",
            "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        ),
        (
            "qwen3-235b-a22b",
            "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml",
        ),
    ],
)
def test_qwen_profiles_accept_non_mamba_scopes(model: str, recipe: str) -> None:
    result = _run_script(
        "scopes/02_moe.sh",
        CLUSTER="oci-hsg",
        MODEL=model,
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    assert recipe in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout


def test_qwen235_profile_leaves_unverified_noncolocated_geometry_unresolved() -> None:
    result = _run_script(
        "scopes/02_moe.sh",
        CLUSTER="ptyche",
        MODEL="qwen3-235b-a22b",
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    unresolved = result.stdout.splitlines()[0]
    assert "TOTAL_NODES" in unresolved
    assert "INFERENCE_NODES" in unresolved


def test_real_mode_fails_on_unresolved_provenance_before_sbatch() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        MODEL="qwen3-235b-a22b",
        TEST_ONLY="0",
    )

    assert result.returncode == 2
    assert "SBATCH:" in result.stdout
    assert "Refusing submission with unresolved fields" in result.stderr


def test_variant_command_uses_configuration_not_graph_scope() -> None:
    result = _run_script(
        "variants/router_preprocess_overlap_true_moe_act_true.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    assert "+policy.megatron_cfg.cuda_graph_modules=\\[moe_router\\,moe_preprocess\\]" in result.stdout
    assert "+policy.megatron_cfg.moe_shared_expert_overlap=true" in result.stdout
    assert "+policy.megatron_cfg.activation_checkpointing=true" in result.stdout
    assert "+policy.megatron_cfg.recompute_granularity=selective" in result.stdout
    assert "+policy.megatron_cfg.recompute_modules=\\[moe_act\\]" in result.stdout


def test_submit_all_smokes_reuses_every_persistent_launcher() -> None:
    result = _run_script(
        "submit_all_smokes.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
        RUN_TAG="unit-test",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("TEST_ONLY: no submission performed") == 41
    assert result.stdout.count("Submitting smoke launcher:") == 41


def test_submit_performance_accepts_explicit_reusable_selection() -> None:
    result = _run_script(
        "submit_performance.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
        RUN_TAG="unit-test",
        PERFORMANCE_SCRIPTS="scopes/00_baseline_no_cg.sh scopes/01_whole_layer.sh",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("TEST_ONLY: no submission performed") == 2
    assert result.stdout.count("Submitting performance launcher:") == 2
    assert "baseline-no-cg" in result.stdout
    assert "whole-layer" in result.stdout


def test_nemorl_integration_gate_uses_bridge_src_layout() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()

    assert (
        "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src" in script
    )


def test_nemorl_integration_gate_uses_official_mcore_environment() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()

    assert "NRL_FORCE_REBUILD_VENVS=true" in script
    assert "uv run --frozen --extra mcore python -m pytest" in script
    assert "export NVTE_CUDA_ARCHS=100" in script
    assert "#SBATCH --time=01:00:00" in script


def test_gb200_profiles_limit_transformer_engine_build_to_sm100() -> None:
    runner = (EXPERIMENT_DIR / "run_scope.sh").read_text()

    assert "/opt/nemo_rl_venv/bin/python" in runner
    assert "uv\n  run\n" not in runner
    assert "MCORE_PYTHONPATH=" in runner
    assert 'PYTHONPATH="${MCORE_PYTHONPATH}" \\' in runner
    assert "Megatron-Bridge/3rdparty/Megatron-LM" in runner
    assert "Megatron-Bridge/src" in runner
    assert 'NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" \\' in runner
    assert 'UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE}" \\' in runner
    assert 'SETUP_COMMAND="${SETUP_COMMAND}" \\' in runner
    assert 'SETUP_COMMAND_ON_WORKERS="${SETUP_COMMAND_ON_WORKERS}" \\' in runner
    assert 'RAY_CLIENT_SERVER_ENABLED="${RAY_CLIENT_SERVER_ENABLED}" \\' in runner
    assert 'RAY_DASHBOARD_ENABLED="${RAY_DASHBOARD_ENABLED}" \\' in runner
    ray_submit = (EXPERIMENT_DIR.parents[2] / "ray.sub").read_text()
    assert "__CONTAINER_LOCAL__" in ray_submit
    assert 'export UV_CACHE_DIR="/root/.cache/uv"' in ray_submit
    assert 'export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1800}"' in ray_submit
    assert "SETUP_COMMAND_ON_WORKERS" in ray_submit
    worker_setup = ray_submit.split("# Workers retry more often", maxsplit=1)[1]
    assert (
        '[[ "$SETUP_COMMAND_ON_WORKERS" == 1 ]]'
        " && [[ -n \"$SETUP_COMMAND_FILE\" ]]" in worker_setup
    )
    for cluster in ("ptyche", "oci-hsg"):
        profile = (EXPERIMENT_DIR / "profiles" / f"{cluster}.env").read_text()
        assert "NVTE_CUDA_ARCHS=100" in profile
        assert "UV_CACHE_DIR_OVERRIDE=__CONTAINER_LOCAL__" in profile
        assert "SETUP_COMMAND=" in profile
        assert "SETUP_COMMAND_ON_WORKERS=0" in profile
        assert "uv run --frozen --extra mcore" not in profile
        assert "--editable" not in profile
        assert "RAY_CLIENT_SERVER_ENABLED=0" in profile
        assert "RAY_DASHBOARD_ENABLED=0" in profile
        assert "--reinstall --no-cache 'ray[default]==2.56.1'" in profile
        assert "--reinstall --no-cache 'dill==0.4.1'" in profile
        assert "--reinstall --no-cache 'numpy==2.5.1'" in profile
        assert "from nemo_rl.algorithms.grpo import MasterConfig" in profile


def test_container_smoke_reuses_official_mcore_environment() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "smoke_nemo_container.sub"
    ).read_text()

    assert "UV_CACHE=/tmp/nemo-rl-uv-cache-" not in script
    assert "export UV_CACHE_DIR=" not in script
    assert "export NVTE_CUDA_ARCHS=100" in script
    assert "export PYTHONPATH=" in script
    assert "Megatron-Bridge/3rdparty/Megatron-LM" in script
    assert "Megatron-Bridge/src" in script
    assert "uv run --frozen --extra mcore" not in script
    assert "--editable" not in script
    assert "--reinstall --no-cache 'ray[default]==2.56.1'" in script
    assert "--reinstall --no-cache 'dill==0.4.1'" in script
    assert "--reinstall --no-cache 'numpy==2.5.1'" in script
    assert "from nemo_rl.algorithms.grpo import MasterConfig" in script
    assert "ray --version" in script
    assert "ray start --head" in script
    assert "ray stop" in script
    assert '"ray"' in script
    assert '"grouped_gemm"' not in script
    assert "src/RL-pr5672-mamba-moe-graph-cache-20260729-d7f1d496f" in script


def test_ray_submission_has_no_singleton_dependency() -> None:
    script = (EXPERIMENT_DIR.parents[2] / "ray.sub").read_text()

    assert "--dependency=singleton" not in script
    assert "RAY_CLIENT_SERVER_ENABLED" in script
    assert "${RAY_CLIENT_SERVER_ARG}" in script
    assert "RAY_DASHBOARD_ENABLED" in script
    assert "${RAY_DASHBOARD_ARGS}" in script


def test_collector_schema_and_wandb_metric_mapping_are_exact() -> None:
    collector = _load_experiment_module("collect_results")
    assert collector.CSV_FIELDS == (
        "scope",
        "job_id",
        "status",
        "step",
        "geometry_key",
        "capture_count",
        "replay_count",
        "cache_hit_count",
        "eviction_count",
        "fallback_count",
        "e2e_step_time",
        "e2e_tokens_per_sec_per_gpu",
        "generation_time",
        "generation_tokens_per_sec_per_gpu",
        "policy_training_time",
        "policy_training_tokens_per_sec_per_gpu",
        "logprob_time",
        "logprob_tokens_per_sec_per_gpu",
        "reward_mean",
        "generation_kl_error",
        "policy_loss",
        "grad_norm",
        "peak_allocated_gib",
        "peak_reserved_gib",
    )
    assert collector.WANDB_METRIC_MAP == {
        "e2e_tokens_per_sec_per_gpu": "performance/tokens_per_sec_per_gpu",
        "generation_tokens_per_sec_per_gpu": (
            "performance/generation_tokens_per_sec_per_gpu"
        ),
        "policy_training_tokens_per_sec_per_gpu": (
            "performance/policy_training_tokens_per_sec_per_gpu"
        ),
        "logprob_tokens_per_sec_per_gpu": (
            "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"
        ),
        "e2e_step_time": "timing/train/total_step_time",
        "generation_time": "timing/train/generation",
        "policy_training_time": "timing/train/policy_training",
        "logprob_time": "timing/train/policy_and_reference_logprobs",
        "reward_mean": "train/reward",
        "generation_kl_error": "train/token_mult_prob_error",
        "policy_loss": "train/loss",
    }
    assert collector.QUALITY_METRICS == (
        "train/reward",
        "train/accuracy",
        "train/token_mult_prob_error",
        "train/loss",
    )


def test_collector_normalizes_nested_local_export_without_network() -> None:
    collector = _load_experiment_module("collect_results")
    row = collector.normalize_record(
        {
            "scope": "attn-moe",
            "job_id": "123",
            "status": "performance:passed",
            "metrics": {
                "_step": 7,
                "cuda_graph/geometry_key": "pp:1",
                "cuda_graph/capture_count": 1,
                "performance/tokens_per_sec_per_gpu": 42.5,
                "performance/generation_tokens_per_sec_per_gpu": 40.0,
                "performance/policy_training_tokens_per_sec_per_gpu": 39.0,
                "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
                    38.0
                ),
                "timing/train/total_step_time": 3.5,
                "timing/train/generation": 1.0,
                "timing/train/policy_training": 1.5,
                "timing/train/policy_and_reference_logprobs": 1.0,
                "train/reward": 0.75,
                "train/token_mult_prob_error": 1.01,
                "train/loss": 0.1,
            },
        }
    )

    assert tuple(row) == collector.CSV_FIELDS
    assert row["step"] == 7
    assert row["geometry_key"] == "pp:1"
    assert row["capture_count"] == 1
    assert row["e2e_tokens_per_sec_per_gpu"] == 42.5
    assert row["logprob_tokens_per_sec_per_gpu"] == 38.0
    assert row["reward_mean"] == 0.75
    assert row["generation_kl_error"] == 1.01
    assert row["policy_loss"] == 0.1


def test_collector_writes_repository_safe_lf_csv(tmp_path: Path) -> None:
    collector = _load_experiment_module("collect_results")
    output = tmp_path / "results.csv"

    collector.write_csv(
        [{"scope": "attn", "job_id": "123", "status": "smoke:submitted"}],
        output,
    )

    assert b"\r\n" not in output.read_bytes()


def _performance_row(
    *,
    scope: str,
    job_id: str,
    step: int,
    multiplier: float,
    correctness_offset: float = 0.0,
    eviction_count: int = 0,
    fallback_count: int = 0,
) -> dict[str, str]:
    """Build one normalized result row with independently checkable metrics."""
    value = float(step)
    return {
        "scope": scope,
        "job_id": job_id,
        "status": "performance:passed",
        "step": str(step),
        "eviction_count": str(eviction_count),
        "fallback_count": str(fallback_count),
        "e2e_step_time": str(value),
        "e2e_tokens_per_sec_per_gpu": str(value * multiplier),
        "generation_time": str(value + 1),
        "generation_tokens_per_sec_per_gpu": str((value + 1) * multiplier),
        "policy_training_time": str(value + 2),
        "policy_training_tokens_per_sec_per_gpu": str((value + 2) * multiplier),
        "logprob_time": str(value + 3),
        "logprob_tokens_per_sec_per_gpu": str((value + 3) * multiplier),
        "reward_mean": str(value / 100 + correctness_offset),
        "generation_kl_error": str(value / 1000 + correctness_offset),
        "policy_loss": str(value / 10000 + correctness_offset),
        "grad_norm": str(value / 10 + correctness_offset),
    }


def test_steady_state_rows_excludes_capture_window() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(4, 21)
    ]

    steady_rows = collector.steady_state_rows(rows)

    assert [row["step"] for row in steady_rows] == [str(step) for step in range(6, 21)]


def test_steady_state_aggregate_groups_runs_and_compares_to_baseline() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(4, 21)
    ]
    rows.extend(
        _performance_row(
            scope="attn",
            job_id="cg-1",
            step=step,
            multiplier=2.0,
            correctness_offset=0.1,
        )
        for step in range(4, 21)
    )
    rows.extend(
        _performance_row(
            scope="attn",
            job_id="cg-2",
            step=step,
            multiplier=1.5,
        )
        for step in range(4, 21)
    )

    aggregated = collector.aggregate_performance(collector.steady_state_rows(rows))

    assert [(row["scope"], row["job_id"]) for row in aggregated] == [
        ("attn", "cg-1"),
        ("attn", "cg-2"),
        ("baseline-no-cg", "baseline-1"),
    ]
    cg_row = aggregated[0]
    for field, median, p95 in (
        ("e2e_step_time", "13", "20"),
        ("e2e_tokens_per_sec_per_gpu", "26", "40"),
        ("generation_time", "14", "21"),
        ("generation_tokens_per_sec_per_gpu", "28", "42"),
        ("policy_training_time", "15", "22"),
        ("policy_training_tokens_per_sec_per_gpu", "30", "44"),
        ("logprob_time", "16", "23"),
        ("logprob_tokens_per_sec_per_gpu", "32", "46"),
    ):
        assert cg_row[f"{field}_median"] == median
        assert cg_row[f"{field}_p95"] == p95
    for field in (
        "e2e_tokens_per_sec_per_gpu",
        "generation_tokens_per_sec_per_gpu",
        "policy_training_tokens_per_sec_per_gpu",
        "logprob_tokens_per_sec_per_gpu",
    ):
        assert cg_row[f"{field}_ratio_to_baseline"] == "2"
    assert cg_row["reward_mean_delta"] == "0.1"
    assert cg_row["generation_kl_error_delta"] == "0.1"
    assert cg_row["policy_loss_delta"] == "0.1"
    assert cg_row["grad_norm_delta"] == "0.1"
    assert cg_row["valid"] == "true"
    assert cg_row["invalid_reason"] == ""


def test_steady_state_aggregate_invalidates_evictions_and_fallbacks() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(6, 21)
    ]
    rows.extend(
        _performance_row(
            scope="attn",
            job_id="cg-eviction",
            step=step,
            multiplier=2.0,
            eviction_count=1 if step >= 8 else 0,
        )
        for step in range(6, 21)
    )
    rows.extend(
        _performance_row(
            scope="attn",
            job_id="cg-fallback",
            step=step,
            multiplier=2.0,
            fallback_count=2 if step >= 12 else 0,
        )
        for step in range(6, 21)
    )

    aggregated = collector.aggregate_performance(rows)
    invalid_rows = {row["job_id"]: row for row in aggregated}

    assert invalid_rows["cg-eviction"]["valid"] == "false"
    assert invalid_rows["cg-eviction"]["invalid_reason"] == "eviction_count=1"
    assert invalid_rows["cg-fallback"]["valid"] == "false"
    assert invalid_rows["cg-fallback"]["invalid_reason"] == "fallback_count=2"


def test_steady_state_aggregate_rejects_missing_baseline() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        _performance_row(
            scope="attn",
            job_id="cg-1",
            step=step,
            multiplier=2.0,
        )
        for step in range(6, 21)
    ]

    aggregate = collector.aggregate_performance(rows)[0]

    assert aggregate["valid"] == "false"
    assert aggregate["invalid_reason"] == "baseline_missing"
    for field in (
        "e2e_tokens_per_sec_per_gpu",
        "generation_tokens_per_sec_per_gpu",
        "policy_training_tokens_per_sec_per_gpu",
        "logprob_tokens_per_sec_per_gpu",
        "reward_mean",
        "generation_kl_error",
        "policy_loss",
        "grad_norm",
    ):
        suffix = "ratio_to_baseline" if "tokens" in field else "delta"
        assert aggregate[f"{field}_{suffix}"] == ""


@pytest.mark.parametrize(
    ("field", "value", "baseline_reason"),
    [
        ("eviction_count", "1", "eviction_count=1"),
        ("fallback_count", "1", "fallback_count=1"),
        ("e2e_step_time", "", "e2e_step_time_missing"),
    ],
)
def test_steady_state_aggregate_rejects_invalid_baseline(
    field: str,
    value: str,
    baseline_reason: str,
) -> None:
    collector = _load_experiment_module("collect_results")
    baseline_rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(6, 21)
    ]
    baseline_rows[2][field] = value
    cg_rows = [
        _performance_row(
            scope="attn",
            job_id="cg-1",
            step=step,
            multiplier=2.0,
        )
        for step in range(6, 21)
    ]

    aggregates = collector.aggregate_performance([*baseline_rows, *cg_rows])
    by_job_id = {aggregate["job_id"]: aggregate for aggregate in aggregates}

    assert by_job_id["baseline-1"]["valid"] == "false"
    assert by_job_id["baseline-1"]["invalid_reason"] == baseline_reason
    assert by_job_id["cg-1"]["valid"] == "false"
    assert (
        by_job_id["cg-1"]["invalid_reason"]
        == "baseline_invalid=baseline-no-cg/baseline-1"
    )
    assert by_job_id["cg-1"]["e2e_tokens_per_sec_per_gpu_ratio_to_baseline"] == ""
    assert by_job_id["cg-1"]["reward_mean_delta"] == ""


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("e2e_step_time", "", "e2e_step_time_missing"),
        ("generation_time", "nan", "generation_time_nonfinite"),
        ("policy_training_time", "inf", "policy_training_time_nonfinite"),
        ("eviction_count", None, "eviction_count_missing"),
        ("fallback_count", None, "fallback_count_missing"),
        ("reward_mean", "", "reward_mean_missing"),
    ],
)
def test_steady_state_aggregate_rejects_incomplete_or_nonfinite_samples(
    field: str,
    value: str | None,
    reason: str,
) -> None:
    collector = _load_experiment_module("collect_results")
    baseline_rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(6, 21)
    ]
    cg_rows = [
        _performance_row(
            scope="attn",
            job_id="cg-1",
            step=step,
            multiplier=2.0,
        )
        for step in range(6, 21)
    ]
    if value is None:
        del cg_rows[4][field]
    else:
        cg_rows[4][field] = value

    aggregates = collector.aggregate_performance([*baseline_rows, *cg_rows])
    cg_aggregate = next(aggregate for aggregate in aggregates if aggregate["job_id"] == "cg-1")

    assert cg_aggregate["valid"] == "false"
    assert cg_aggregate["invalid_reason"] == reason
    for throughput_field in (
        "e2e_tokens_per_sec_per_gpu",
        "generation_tokens_per_sec_per_gpu",
        "policy_training_tokens_per_sec_per_gpu",
        "logprob_tokens_per_sec_per_gpu",
    ):
        assert cg_aggregate[f"{throughput_field}_ratio_to_baseline"] == ""
    for correctness_field in (
        "reward_mean",
        "generation_kl_error",
        "policy_loss",
        "grad_norm",
    ):
        assert cg_aggregate[f"{correctness_field}_delta"] == ""


def test_report_renders_overlay_provenance_steady_state_and_raw_failures() -> None:
    renderer = _load_experiment_module("render_report")
    report = renderer.render_html(
        [
            {
                "scope": "attn",
                "job_id": "raw-failure",
                "status": "performance:invalid",
                "step": "8",
                "fallback_count": "1",
            }
        ],
        nemo_rl_sha="nemo-sha",
        bridge_sha="bridge-sha",
        container_sha256="container-sha",
        te_version="2.15.0+42b84005",
        te_source_commit="e707aa46869dc2aec08dfea25402e97a61d49fef",
        te_overlay_sha256="39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b",
    )

    for value in (
        "2.15.0+42b84005",
        "e707aa46869dc2aec08dfea25402e97a61d49fef",
        "39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b",
        "container-sha",
        "6–20",
        "raw-failure",
    ):
        assert value in report
    assert '<section id="steady-state-performance">' in report
    assert '<section id="correctness-deltas">' in report
    assert '<section id="failures">' in report


def test_report_has_required_sections_scope_labels_and_verified_status() -> None:
    renderer = _load_experiment_module("render_report")
    assert renderer.DEFAULT_MCORE_SHA == "100047b517ea91526dc465448fcb3b37b2598388"
    report = renderer.render_html(
        [
            {
                "scope": "baseline-no-cg",
                "job_id": "1",
                "status": "smoke:passed",
            },
            {
                "scope": "whole-layer",
                "job_id": "2",
                "status": "performance:passed",
                "e2e_step_time": "1.5",
            },
            {
                "scope": "moe-overlap-true-moe-act-false",
                "job_id": "3",
                "status": "accuracy:failed",
                "reward_mean": "0.25",
            },
        ],
        te_version="2.15.0+42b84005",
        te_source_commit="e707aa46869dc2aec08dfea25402e97a61d49fef",
        te_overlay_sha256=TE_FP64_WEAKREF_SHA256,
    )

    for section in (
        "Correctness",
        "Smoke",
        "Performance",
        "Accuracy",
        "Failures",
        "Provenance",
    ):
        assert f"<h2>{section}</h2>" in report
    assert "No-CG baseline (CUDA graphs disabled)" in report
    assert "TE whole-layer capture (empty module list)" in report
    assert "configuration variant; graph scope unchanged" in report
    assert "2471224" in report and "66 passed" in report
    assert "2471343" in report and "29 + 3 passed" in report
    assert "2471570" in report and "38 + 3 passed" in report
    assert "2471681" in report and "43 + 23 passed" in report
    assert "2471988" in report and "packed Mamba parity passed" in report
    assert "MoE 5→3→5 passed" in report
    assert "74.33s" in report and "6.96s" in report and "82.78s" in report
    assert "100047b517ea91526dc465448fcb3b37b2598388" in report
    assert "37 host tests + Pyrefly passed" in report
    assert "Task 7" in report and "Slurm 2472646" in report
    assert "138 passed" in report
    assert renderer.DEFAULT_MODEL_SNAPSHOT in report
    assert renderer.DEFAULT_TOKENIZER_SNAPSHOT in report
    assert "__REQUIRED_*_MODEL_SNAPSHOT__" not in report


def test_checked_in_report_is_static_and_has_all_sections() -> None:
    renderer = _load_experiment_module("render_report")
    report_path = (
        EXPERIMENT_DIR.parents[0]
        / "results"
        / "mamba_moe_te_graph_20260729_report.html"
    )
    report = report_path.read_text()

    assert report.startswith("<!doctype html>")
    for section_id in (
        "correctness",
        "smoke",
        "performance",
        "accuracy",
        "failures",
        "provenance",
    ):
        assert f'<section id="{section_id}">' in report
    assert "2471988" in report
    assert renderer.DEFAULT_MCORE_SHA in report
    assert "Slurm 2472646" in report
    assert "138 passed" in report
    assert renderer.DEFAULT_MODEL_SNAPSHOT in report
    assert renderer.DEFAULT_TOKENIZER_SNAPSHOT in report
