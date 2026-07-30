import importlib.util
import itertools
import os
import re
import subprocess
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
    spec.loader.exec_module(module)
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


def test_test_only_reports_unresolved_provenance_and_never_submits() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    assert "UNRESOLVED:" in result.stdout
    assert "COMMAND:" in result.stdout
    assert "SBATCH:" in result.stdout
    assert (
        "examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml"
        in result.stdout
    )
    assert "policy.megatron_cfg.cuda_graph_modules=\\[attn\\]" in result.stdout
    assert "policy.megatron_cfg.cuda_graph_warmup_steps=3" in result.stdout
    assert "policy.megatron_cfg.cuda_graph_max_cached_schedules=2" in result.stdout
    assert "policy.megatron_cfg.cuda_graph_max_packed_seqs=16" in result.stdout
    assert "checkpointing.enabled=false" in result.stdout
    assert "logger.wandb.project=sna-cg-study" in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout


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
    assert "cuda_graph_modules=\\[moe_router\\,moe_preprocess\\]" in result.stdout
    assert "moe_shared_expert_overlap=true" in result.stdout
    assert "activation_checkpointing=true" in result.stdout
    assert "recompute_granularity=selective" in result.stdout
    assert "recompute_modules=\\[moe_act\\]" in result.stdout


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
        ]
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
    assert "Task 7" in report and "uncommitted / in progress" in report


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
    assert "uncommitted / in progress" in report
