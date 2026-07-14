import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
SUBMITTER = EXPERIMENT_DIR / "submit_nemo2606_2n4g_factorial.sh"
OFFICIAL_SUBMITTER = EXPERIMENT_DIR / "submit_nemo2606_4n4g_performance.sh"
DIRECT_AB_SUBMITTER = EXPERIMENT_DIR / "submit_cutedsl_ab_replicates.sh"
MATRIX_PAYLOAD = EXPERIMENT_DIR / "run_cutedsl_matrix.sbatch"
PROFILE_LOADER = EXPERIMENT_DIR / "lib/cluster_profile.sh"
RAY_SUB = PROJECT_ROOT / "ray.sub"
RECIPE = (
    PROJECT_ROOT
    / "examples/configs/recipes/llm/performance"
    / "grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-factorial.yaml"
)
OFFICIAL_RECIPE = (
    PROJECT_ROOT
    / "examples/configs/recipes/llm/performance"
    / "grpo-qwen3-30ba3b-4n4g-megatron-mxfp8-cutedsl.yaml"
)
OFFICIAL_BASE_RECIPE = (
    PROJECT_ROOT
    / "examples/configs/recipes/llm/performance"
    / "grpo-qwen3-30ba3b-4n4g.yaml"
)

register_omegaconf_resolvers()


def _run_timing_summarizer(
    tmp_path: Path,
    *,
    on_token_scale: float,
    timing_order: tuple[str, ...] = ("on", "off"),
) -> subprocess.CompletedProcess[str]:
    source = MATRIX_PAYLOAD.read_text()
    summarizer = source.split("# CUTEDSL_TIMING_SUMMARIZER_START\n", 1)[1].split(
        "# CUTEDSL_TIMING_SUMMARIZER_END", 1
    )[0]
    result_dir = tmp_path / "results"
    resolved_metric_names = {"train/total_num_tokens": "train/total_num_tokens"}
    for order_index, arm in enumerate(timing_order):
        arm_dir = result_dir / "timing" / f"{order_index}-{arm}"
        arm_dir.mkdir(parents=True)
        scale = on_token_scale if arm == "on" else 1.0
        rows = []
        for offset, step in enumerate((6, 7, 8)):
            total_tokens = float(round((1_000_000 + offset * 10_000) * scale))
            valid_tokens = float(round((900_000 + offset * 10_000) * scale))
            rows.append(
                {
                    "step": step,
                    "total_num_tokens": total_tokens,
                    "global_valid_toks": valid_tokens,
                    "mean_prompt_length": 128.0 + offset,
                    "num_valid_samples": 2048.0,
                    "total_turns": 2048.0,
                    "policy_training_tokens_per_sec_per_gpu": 5000.0,
                }
            )
        raw = {
            "run_id": "job-a",
            "arm": arm,
            "order_index": order_index,
            "policy_training_seconds": [80.0, 81.0, 82.0],
            "resolved_metric_names": resolved_metric_names,
            "measured_step_workload": rows,
        }
        (arm_dir / "raw_timing.json").write_text(json.dumps(raw))
    (result_dir / "benchmark_manifest.json").write_text(json.dumps({}))
    env = os.environ.copy()
    env.update(
        {
            "CONTAINER_RESULT_DIR": str(result_dir),
            "TIMING_ORDER": ",".join(timing_order),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", summarizer],
        capture_output=True,
        text=True,
        env=env,
    )


def _run_kernel_attribution_fixture(
    tmp_path: Path,
    *,
    on: str,
    off: str,
    off_moe_grouped_gemm: bool = True,
    off_op_fuser: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    """Execute the embedded kernel-attribution program on bounded fixtures."""
    source = MATRIX_PAYLOAD.read_text()
    attribution = source.split("# CUTEDSL_KERNEL_ATTRIBUTION_START\n", 1)[1].split(
        "# CUTEDSL_KERNEL_ATTRIBUTION_END", 1
    )[0]
    result_dir = tmp_path / "results"
    for order_index, (arm, evidence) in enumerate((("on", on), ("off", off))):
        profile_dir = result_dir / "profiles" / f"{order_index}-{arm}"
        profile_dir.mkdir(parents=True)
        (profile_dir / "kernel_evidence.txt").write_text(evidence)
    config_evidence = {
        "on": {
            "policy.megatron_cfg.moe_grouped_gemm": True,
            "policy.megatron_cfg.use_transformer_engine_op_fuser": True,
        },
        "off": {
            "policy.megatron_cfg.moe_grouped_gemm": off_moe_grouped_gemm,
            "policy.megatron_cfg.use_transformer_engine_op_fuser": off_op_fuser,
        },
    }
    manifest = {
        "available_arms": ["on", "off"],
        "fixed_config_evidence": config_evidence,
        "feature_context": "g0a0",
        "full_cg_enabled": False,
        "a2a_enabled": False,
    }
    (result_dir / "benchmark_manifest.json").write_text(json.dumps(manifest))

    result = subprocess.run(
        [sys.executable, "-c", attribution, str(result_dir)],
        capture_output=True,
        text=True,
    )
    output_path = result_dir / "kernel_attribution.json"
    output: dict[str, object] = (
        json.loads(output_path.read_text()) if output_path.is_file() else {}
    )
    return result, output


def _actual_cudnn_fused_kernel_evidence() -> str:
    """Return the exact object-suffixed kernel-name shape seen in job 2369786."""
    return "\n".join(
        (
            "kernel_cutlass_kernel_cudnngrouped_gemm_"
            "BlockScaledMoEGroupedGemmQuantKernel_object_at_0x1",
            "kernel_cutlass_kernel_cudnngrouped_gemm_"
            "BlockScaledMoEGroupedGemmGluBiasKernel_object_at_0x2",
            "kernel_cutlass_kernel_cudnngrouped_gemm_"
            "BlockScaledMoEGroupedGemmDgluDbiasKernel_object_at_0x3",
        )
    )


def test_kernel_matchers_accept_cudnn_object_suffix_and_reject_off_arm(
    tmp_path: Path,
) -> None:
    result, attribution = _run_kernel_attribution_fixture(
        tmp_path,
        on=_actual_cudnn_fused_kernel_evidence(),
        off="nvjet_sm100_128x128",
    )

    assert result.returncode == 0, result.stderr
    arms = attribution["arms"]
    assert isinstance(arms, dict)
    assert arms["on"]["fused_glu_match_count"] == 1
    assert arms["on"]["fused_dglu_match_count"] == 1
    assert arms["on"]["fused_quant_match_count"] == 1
    assert arms["on"]["fused_grouped_gemm_match_count"] == 3
    assert arms["off"]["fused_glu_match_count"] == 0
    assert arms["off"]["fused_dglu_match_count"] == 0
    assert arms["off"]["fused_quant_match_count"] == 0
    assert arms["off"]["fused_grouped_gemm_match_count"] == 0
    assert arms["off"]["baseline_expert_gemm_match_count"] == 1


@pytest.mark.parametrize(
    ("off_moe_grouped_gemm", "off_op_fuser", "failure"),
    [
        (False, True, "OFF grouped GEMM config evidence is not true"),
        (True, False, "OFF op fuser config evidence is not true"),
    ],
)
def test_off_baseline_attribution_requires_fixed_config_predicates(
    tmp_path: Path,
    off_moe_grouped_gemm: bool,
    off_op_fuser: bool,
    failure: str,
) -> None:
    result, attribution = _run_kernel_attribution_fixture(
        tmp_path,
        on=_actual_cudnn_fused_kernel_evidence(),
        off="nvjet_sm100_128x128",
        off_moe_grouped_gemm=off_moe_grouped_gemm,
        off_op_fuser=off_op_fuser,
    )

    assert result.returncode != 0
    assert failure in result.stderr
    arms = attribution["arms"]
    assert isinstance(arms, dict)
    assert arms["off"]["baseline_expert_gemm_match_count"] == 0


def test_kernel_attribution_rejects_fused_kernel_in_off_arm(tmp_path: Path) -> None:
    result, attribution = _run_kernel_attribution_fixture(
        tmp_path,
        on=_actual_cudnn_fused_kernel_evidence(),
        off=(
            "nvjet_sm100_128x128\nBlockScaledMoEGroupedGemmGluBiasKernel_object_at_0x4"
        ),
    )

    assert result.returncode != 0
    assert "OFF fused GLU kernel signature must be absent" in result.stderr
    assert attribution["passed"] is False


def test_kernel_attribution_rejects_on_arm_without_fused_quant(
    tmp_path: Path,
) -> None:
    on = _actual_cudnn_fused_kernel_evidence().replace(
        "BlockScaledMoEGroupedGemmQuantKernel_object_at_0x1", ""
    )
    result, attribution = _run_kernel_attribution_fixture(
        tmp_path,
        on=on,
        off="nvjet_sm100_128x128",
    )

    assert result.returncode != 0
    assert "ON fused quant kernel signature was not found" in result.stderr
    assert attribution["passed"] is False


def _run_functional_summarizer(
    tmp_path: Path,
    offload_sequence: int,
    *,
    cgroup_memory_peak_gib: str = "unavailable",
    cgroup_memory_max_gib: str = "unavailable",
    ray_logs: dict[str, str] | None = None,
    constant_overrides: dict[str, int] | None = None,
    evidence_suffix: str = "",
    training_gpu_count: int = 8,
) -> subprocess.CompletedProcess[str]:
    source = MATRIX_PAYLOAD.read_text()
    summarizer = source.split("# CUTEDSL_FUNCTIONAL_SUMMARIZER_START\n", 1)[1].split(
        "# CUTEDSL_FUNCTIONAL_SUMMARIZER_END", 1
    )[0]
    for constant, value in (constant_overrides or {}).items():
        summarizer = re.sub(
            rf"^{re.escape(constant)} = .+$",
            f"{constant} = {value}",
            summarizer,
            count=1,
            flags=re.MULTILINE,
        )
    result_dir = tmp_path / "results"
    arm_dir = result_dir / "functional" / "0-on"
    ray_log_dir = tmp_path / "ray-logs"
    arm_dir.mkdir(parents=True)
    ray_log_dir.mkdir()
    metric_steps = {str(step): float(step + 1) for step in range(3)}
    metrics = {
        metric: metric_steps
        for metric in (
            "timing/train/total_step_time",
            "timing/train/generation",
            "timing/train/get_logprobs",
            "timing/train/policy_training",
            "timing/train/prepare_for_generation/transfer_and_update_weights",
        )
    }
    (arm_dir / "metrics.json").write_text(json.dumps(metrics))
    evidence_lines = ["kernel=GroupedGemmGluSm100"]
    evidence_lines.extend(
        "event=megatron_policy_offload_memory phase=after_completion "
        f"global_rank={rank} offload_sequence={offload_sequence} "
        f"cgroup_memory_peak_gib={cgroup_memory_peak_gib} "
        f"cgroup_memory_max_gib={cgroup_memory_max_gib}{evidence_suffix}"
        for rank in range(training_gpu_count)
    )
    (arm_dir / "grpo.log").write_text("\n".join(evidence_lines) + "\n")
    for relative_path, contents in (ray_logs or {}).items():
        ray_path = ray_log_dir / relative_path
        ray_path.parent.mkdir(parents=True, exist_ok=True)
        ray_path.write_text(contents)
    (result_dir / "benchmark_manifest.json").write_text("{}\n")
    env = os.environ.copy()
    env.update(
        {
            "CONTAINER_RESULT_DIR": str(result_dir),
            "FUNCTIONAL_UPDATES": "3",
            "RAY_CLUSTER_LOG_DIR": str(ray_log_dir),
            "TRAINING_GPU_COUNT": str(training_gpu_count),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", summarizer],
        env=env,
        capture_output=True,
        text=True,
    )


def test_multinode_recipe_has_ep8_and_two_local_microbatches() -> None:
    config = OmegaConf.to_container(load_config(RECIPE), resolve=True)
    assert isinstance(config, dict)
    policy = config["policy"]
    megatron = policy["megatron_cfg"]
    world_size = config["cluster"]["num_nodes"] * config["cluster"]["gpus_per_node"]
    model_parallel_size = (
        megatron["tensor_model_parallel_size"]
        * megatron["pipeline_model_parallel_size"]
        * megatron["context_parallel_size"]
    )
    data_parallel_size = world_size // model_parallel_size
    local_microbatches = policy["train_global_batch_size"] // (
        policy["train_micro_batch_size"] * data_parallel_size
    )

    assert policy["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert config["cluster"] == {
        **config["cluster"],
        "num_nodes": 2,
        "gpus_per_node": 4,
    }
    assert config["cluster"]["segment_size"] == 2
    assert megatron["expert_model_parallel_size"] == 8
    assert megatron["expert_tensor_parallel_size"] == 1
    assert policy["train_global_batch_size"] == 16
    assert policy["train_micro_batch_size"] == 1
    assert local_microbatches == 2
    assert policy["dynamic_batching"]["enabled"] is False
    assert policy["sequence_packing"]["enabled"] is False
    assert policy["max_total_sequence_length"] == 1024
    assert config["grpo"]["num_prompts_per_step"] == 8
    assert config["grpo"]["num_generations_per_prompt"] == 2
    assert (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        == policy["train_global_batch_size"]
    )


def test_official_performance_recipe_preserves_workload_and_enables_policy_mxfp8() -> (
    None
):
    config = OmegaConf.to_container(load_config(OFFICIAL_RECIPE), resolve=True)
    assert isinstance(config, dict)
    policy = config["policy"]
    megatron = policy["megatron_cfg"]

    assert config["cluster"]["num_nodes"] == 4
    assert config["cluster"]["gpus_per_node"] == 4
    assert config["cluster"]["segment_size"] == 4
    assert config["grpo"]["num_prompts_per_step"] == 64
    assert config["grpo"]["num_generations_per_prompt"] == 32
    assert config["grpo"]["val_period"] == 10
    assert config["grpo"]["val_at_start"] is False
    assert config["grpo"]["val_at_end"] is False
    assert policy["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert policy["train_global_batch_size"] == 2048
    assert policy["train_micro_batch_size"] == 1
    assert policy["logprob_batch_size"] == 2
    assert policy["max_total_sequence_length"] == 4096
    assert policy["dynamic_batching"]["enabled"] is False
    assert policy["sequence_packing"]["enabled"] is True

    assert megatron["tensor_model_parallel_size"] == 1
    assert megatron["pipeline_model_parallel_size"] == 1
    assert megatron["context_parallel_size"] == 1
    assert megatron["expert_tensor_parallel_size"] == 1
    assert megatron["expert_model_parallel_size"] == 16
    assert megatron["moe_grouped_gemm"] is True
    assert megatron["moe_router_dtype"] == "fp32"
    assert megatron["use_transformer_engine_op_fuser"] is True
    assert megatron["moe_mlp_glu_interleave_size"] == 32
    assert megatron["fp8_cfg"] == {
        **megatron["fp8_cfg"],
        "enabled": True,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "fp8_param": False,
    }
    assert megatron["env_vars"]["PYTORCH_CUDA_ALLOC_CONF"] == (
        "expandable_segments:False"
    )
    assert megatron["env_vars"]["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == "1"

    vllm = policy["generation"]["vllm_cfg"]
    assert vllm["precision"] == "bfloat16"
    assert vllm["tensor_parallel_size"] == 1
    assert vllm["env_vars"]["VLLM_COMPILE_CACHE_SAVE_FORMAT"] == "unpacked"
    assert "VLLM_USE_STANDALONE_COMPILE" not in vllm["env_vars"]


def test_official_performance_overlay_has_only_reviewed_deviations() -> None:
    base = OmegaConf.to_container(load_config(OFFICIAL_BASE_RECIPE), resolve=True)
    overlay = OmegaConf.to_container(load_config(OFFICIAL_RECIPE), resolve=True)
    assert isinstance(base, dict)
    assert isinstance(overlay, dict)
    for config in (base, overlay):
        assert (
            config["grpo"]["val_period"],
            config["grpo"]["val_at_start"],
            config["grpo"]["val_at_end"],
        ) == (10, False, False)

    def difference_paths(left: object, right: object, path: str = "") -> set[str]:
        if isinstance(left, dict) and isinstance(right, dict):
            differences = set()
            for key in set(left) | set(right):
                child_path = f"{path}.{key}" if path else key
                if key not in left or key not in right:
                    differences.add(child_path)
                else:
                    differences.update(
                        difference_paths(left[key], right[key], child_path)
                    )
            return differences
        return set() if left == right else {path}

    assert difference_paths(base, overlay) == {
        "checkpointing.checkpoint_dir",
        "logger.log_dir",
        "logger.wandb.name",
        "policy.generation.vllm_cfg.env_vars",
        "policy.megatron_cfg.cuda_graph_impl",
        "policy.megatron_cfg.cuda_graph_use_single_mempool",
        "policy.megatron_cfg.cuda_graph_warmup_steps",
        "policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        "policy.megatron_cfg.fp8_cfg.enabled",
        "policy.megatron_cfg.fp8_cfg.fp8_recipe",
        "policy.megatron_cfg.moe_mlp_glu_interleave_size",
        "policy.megatron_cfg.moe_router_dtype",
        "policy.megatron_cfg.use_transformer_engine_op_fuser",
    }


def test_matrix_disables_validation_only_for_performance_arms() -> None:
    source = MATRIX_PAYLOAD.read_text()
    start_marker = "# NEMO2606_PERFORMANCE_VALIDATION_OVERRIDES_START"
    end_marker = "# NEMO2606_PERFORMANCE_VALIDATION_OVERRIDES_END"
    assert start_marker in source
    assert end_marker in source
    performance_block = source.split(start_marker, 1)[1].split(end_marker, 1)[0]
    assert 'if [[ "${FUNCTIONAL_GATE}" == "0" ]]; then' in performance_block
    assert '"grpo.val_period=0"' in performance_block
    assert '"grpo.val_at_start=false"' in performance_block
    assert '"grpo.val_at_end=false"' in performance_block

    initializer = source.split("COMMON_OVERRIDES=(", 1)[1].split(")\n", 1)[0]
    assert "grpo.val_period" not in initializer
    assert "grpo.val_at_start" not in initializer
    assert "grpo.val_at_end" not in initializer

    functional = OmegaConf.to_container(load_config(OFFICIAL_RECIPE), resolve=True)
    timing = OmegaConf.to_container(
        parse_hydra_overrides(
            load_config(OFFICIAL_RECIPE),
            [
                "grpo.val_period=0",
                "grpo.val_at_start=false",
                "grpo.val_at_end=false",
            ],
        ),
        resolve=True,
    )
    assert isinstance(functional, dict)
    assert isinstance(timing, dict)
    assert (
        functional["grpo"]["val_period"],
        functional["grpo"]["val_at_start"],
        functional["grpo"]["val_at_end"],
    ) == (10, False, False)
    assert (
        timing["grpo"]["val_period"],
        timing["grpo"]["val_at_start"],
        timing["grpo"]["val_at_end"],
    ) == (0, False, False)


def test_official_performance_recipe_accepts_full_iteration_overrides() -> None:
    config = parse_hydra_overrides(
        load_config(OFFICIAL_RECIPE),
        [
            "policy.megatron_cfg.cuda_graph_impl=full_iteration",
            "policy.megatron_cfg.cuda_graph_warmup_steps=3",
            "policy.megatron_cfg.cuda_graph_use_single_mempool=true",
        ],
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    megatron = resolved["policy"]["megatron_cfg"]
    assert megatron["cuda_graph_impl"] == "full_iteration"
    assert megatron["cuda_graph_warmup_steps"] == 3
    assert megatron["cuda_graph_use_single_mempool"] is True


def test_multinode_recipe_uses_unpacked_vllm_compile_cache() -> None:
    config = OmegaConf.to_container(load_config(RECIPE), resolve=True)
    assert isinstance(config, dict)

    vllm_env = config["policy"]["generation"]["vllm_cfg"]["env_vars"]
    assert vllm_env["VLLM_COMPILE_CACHE_SAVE_FORMAT"] == "unpacked"
    assert "VLLM_USE_STANDALONE_COMPILE" not in vllm_env


def test_submitter_launches_real_two_node_ray_sub_cluster() -> None:
    source = SUBMITTER.read_text()
    required = (
        'readonly RAY_SUB="${REPO_ROOT}/ray.sub"',
        '"--nodes=${BENCHMARK_NUM_NODES}"',
        '"CONTAINER=${CUTEDSL_IMAGE}"',
        '"GPUS_PER_NODE=${BENCHMARK_GPUS_PER_NODE}"',
        '"COMMAND=exec bash ${MATRIX_PAYLOAD}"',
        '"CUTEDSL_BENCHMARK_EXISTING_RAY=1"',
        '"RAY_LOG_SYNC_FREQUENCY=5"',
        '"BASE_LOG_DIR=${RAY_LOG_ROOT}"',
        '"MOUNTS=${RAY_MOUNTS}"',
        '"SETUP_COMMAND=${RAY_SETUP_COMMAND}"',
        'GIT_COMMON_DIR=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)',
        'RAY_MOUNTS+=",${GIT_COMMON_DIR}:${GIT_COMMON_DIR}"',
        '"${RAY_SUB}"',
    )
    for fragment in required:
        assert fragment in source, fragment
    assert (
        '"${MATRIX_PAYLOAD}"'
        not in re.sub(r'"COMMAND=exec bash \$\{MATRIX_PAYLOAD\}"', "", source).split(
            "job_id=$(sbatch", 1
        )[1]
    )


def _capture_submitter_exports(
    tmp_path: Path,
    *,
    functional_gate: bool,
) -> list[dict[str, str]]:
    mock_bin = tmp_path / ("functional-bin" if functional_gate else "factorial-bin")
    mock_bin.mkdir()
    calls_path = tmp_path / (
        "functional-exports.jsonl" if functional_gate else "factorial-exports.jsonl"
    )
    mock_sbatch = mock_bin / "sbatch"
    mock_sbatch.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

payload_arg = next(arg for arg in sys.argv[1:] if arg.startswith("--export-file="))
payload = {}
for entry in Path(payload_arg.split("=", 1)[1]).read_bytes().split(b"\\0"):
    if entry:
        key, value = entry.decode().split("=", 1)
        payload[key] = value
record = {
    "functional_gate": payload.get("NEMO2606_FUNCTIONAL_GATE", "0"),
    "uv_no_editable": payload.get("UV_NO_EDITABLE", "<missing>"),
}
with Path(os.environ["MOCK_SBATCH_CALLS"]).open("a") as output:
    output.write(json.dumps(record) + "\\n")
print("mock-job")
"""
    )
    mock_sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
            "NEMO2606_FUNCTIONAL_GATE": "1" if functional_gate else "0",
            "UV_NO_EDITABLE": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(SUBMITTER), "--test-only"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in calls_path.read_text().splitlines()]


def test_submitter_forces_noneditable_local_packages_before_ray_start(
    tmp_path: Path,
) -> None:
    functional_exports = _capture_submitter_exports(tmp_path, functional_gate=True)
    factorial_exports = _capture_submitter_exports(tmp_path, functional_gate=False)

    assert len(functional_exports) == 1
    assert len(factorial_exports) == 6
    assert {
        export["uv_no_editable"] for export in functional_exports + factorial_exports
    } == {"1"}


def test_direct_ab_submitter_forces_noneditable_local_packages(
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "direct-bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "direct-exports.jsonl"
    mock_sbatch = mock_bin / "sbatch"
    mock_sbatch.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

payload_arg = next(arg for arg in sys.argv[1:] if arg.startswith("--export-file="))
payload = {}
for entry in Path(payload_arg.split("=", 1)[1]).read_bytes().split(b"\\0"):
    if entry:
        key, value = entry.decode().split("=", 1)
        payload[key] = value
with Path(os.environ["MOCK_SBATCH_CALLS"]).open("a") as output:
    output.write(json.dumps({"uv_no_editable": payload.get("UV_NO_EDITABLE")}) + "\\n")
print("mock-direct")
"""
    )
    mock_sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
            "UV_NO_EDITABLE": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(DIRECT_AB_SUBMITTER), "--test-only"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 3
    assert {call["uv_no_editable"] for call in calls} == {"1"}


def test_ray_sub_gates_driver_on_all_workers_and_shared_setup() -> None:
    source = RAY_SUB.read_text()
    assert "NUM_ACTORS=$((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))" in source
    assert 'if [[ "$worker_units" -eq "$NUM_ACTORS" ]]' in source
    assert "_num_workers=$((SLURM_JOB_NUM_NODES - 1))" in source
    assert "--ntasks=$_num_workers" in source
    assert 'bash "$DRIVER_COMMAND_FILE"' in source
    assert source.count('bash "$SETUP_COMMAND_FILE"') == 2
    assert (
        ".shared_fs_canary not visible; LOG_DIR must be on a shared filesystem"
        in source
    )


def _ray_template_block(start_marker: str, end_marker: str) -> str:
    source = RAY_SUB.read_text()
    assert start_marker in source
    assert end_marker in source
    return source.split(start_marker, 1)[1].split(end_marker, 1)[0].replace(r"\$", "$")


def test_ray_sub_runs_failure_hook_before_ended_cleanup() -> None:
    source = RAY_SUB.read_text()
    driver_failure_block = source.split('bash "$DRIVER_COMMAND_FILE"', 1)[1].split(
        "else\n  # Interactive", 1
    )[0]

    assert 'FAILURE_COMMAND_FILE=""' in source
    assert 'touch "$LOG_DIR/DRIVER_FAILED"' in driver_failure_block
    assert "FAILURE_DIAGNOSTIC_DONE_0" in driver_failure_block
    assert (
        r"FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS="
        r"\${FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS:-60}" in driver_failure_block
    )
    assert r"^([1-9]|[1-5][0-9]|60)$" in driver_failure_block
    assert "run-failure-command-until()" in driver_failure_block
    assert 'bash "$FAILURE_COMMAND_FILE" &' in driver_failure_block
    assert r'kill -0 "\$command_pid"' in driver_failure_block
    assert r'kill -TERM "\$command_pid"' in driver_failure_block
    assert r'kill -KILL "\$command_pid"' in driver_failure_block
    assert r'wait "\$command_pid"' in driver_failure_block
    assert r"FAILURE_DIAGNOSTIC_DEADLINE=\$((" in driver_failure_block
    assert r"FAILURE_DIAGNOSTIC_COLLECTION_DEADLINE=\$((" in driver_failure_block
    assert "export FAILURE_DIAGNOSTIC_MERGE=1" in driver_failure_block
    assert "SECONDS < FAILURE_DIAGNOSTIC_COLLECTION_DEADLINE" in driver_failure_block
    assert "SECONDS <=" not in driver_failure_block
    aggregate_deadline = driver_failure_block.index(r"FAILURE_DIAGNOSTIC_DEADLINE=\$((")
    assert aggregate_deadline < driver_failure_block.index(
        'touch "$LOG_DIR/DRIVER_FAILED"'
    )
    assert driver_failure_block.count("run-failure-command-until") == 3
    assert driver_failure_block.index('touch "$LOG_DIR/DRIVER_FAILED"') < (
        driver_failure_block.index('touch "$LOG_DIR/FAILURE_DIAGNOSTIC_DONE_0"')
    )
    assert driver_failure_block.index(
        'touch "$LOG_DIR/FAILURE_DIAGNOSTIC_DONE_0"'
    ) < driver_failure_block.index("export FAILURE_DIAGNOSTIC_MERGE=1")
    assert driver_failure_block.index("export FAILURE_DIAGNOSTIC_MERGE=1") < (
        driver_failure_block.index('touch "$LOG_DIR/ENDED"')
    )
    ended_line = next(
        line
        for line in driver_failure_block.splitlines()
        if 'touch "$LOG_DIR/ENDED"' in line
    )
    assert ended_line.endswith("2>/dev/null || true")


def test_ray_sub_worker_failure_sidecar_is_one_shot_and_non_destructive() -> None:
    source = RAY_SUB.read_text()
    worker_block = source.split("worker_cmd=$(cat <<EOF", 1)[1].split("\nEOF\n)", 1)[0]

    assert "failure-diagnostic-sidecar()" in worker_block
    assert '[[ -f "$LOG_DIR/DRIVER_FAILED" ]]' in worker_block
    assert r"FAILURE_DIAGNOSTIC_NODE_INDEX=\$((SLURM_PROCID + 1))" in worker_block
    assert 'if bash "$FAILURE_COMMAND_FILE"; then' in worker_block
    assert r"FAILURE_DIAGNOSTIC_DONE_\${FAILURE_DIAGNOSTIC_NODE_INDEX}" in worker_block
    sidecar_start = worker_block.index("failure-diagnostic-sidecar &")
    assert sidecar_start < worker_block.index('ray start --address "$ip_head"')
    sidecar_block = worker_block.split("failure-diagnostic-sidecar()", 1)[1].split(
        "failure-diagnostic-sidecar &", 1
    )[0]
    assert "exit-dramatically" not in sidecar_block


@pytest.mark.parametrize(
    ("exit_code", "hook_enabled"),
    ((0, True), (17, True), (17, False)),
)
def test_generated_head_failure_hook_is_opt_in_and_ordered(
    tmp_path: Path, exit_code: int, hook_enabled: bool
) -> None:
    hook = _ray_template_block("# RAY_FAILURE_HOOK_START\n", "# RAY_FAILURE_HOOK_END\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    invocations = tmp_path / "invocations"
    failure_command = tmp_path / "failure-command.sh"
    failure_command.write_text(
        "#!/bin/bash\n"
        f"printf '%s:%s\\n' \"$FAILURE_DIAGNOSTIC_NODE_INDEX\" "
        f'"${{FAILURE_DIAGNOSTIC_MERGE:-0}}" >> {shlex.quote(str(invocations))}\n'
        'if [[ -z "${FAILURE_DIAGNOSTIC_MERGE:-}" ]]; then\n'
        f"  touch {shlex.quote(str(log_dir / 'FAILURE_DIAGNOSTIC_DONE_1'))}\n"
        "fi\n"
        "exit 9\n"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    f"LOG_DIR={shlex.quote(str(log_dir))}",
                    "FAILURE_COMMAND_FILE="
                    + (shlex.quote(str(failure_command)) if hook_enabled else "''"),
                    "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=2",
                    "SLURM_JOB_NUM_NODES=2",
                    f"exit_code={exit_code}",
                    hook,
                    'exit "$exit_code"',
                )
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == exit_code, result.stderr
    if exit_code == 0 or not hook_enabled:
        assert not invocations.exists()
        assert not (log_dir / "DRIVER_FAILED").exists()
        assert not list(log_dir.glob("FAILURE_DIAGNOSTIC_DONE_*"))
        assert not (log_dir / "ENDED").exists()
    else:
        assert invocations.read_text().splitlines() == ["0:0", "0:1"]
        assert (log_dir / "DRIVER_FAILED").is_file()
        assert not (log_dir / "FAILURE_DIAGNOSTIC_DONE_0").exists()
        assert (log_dir / "FAILURE_DIAGNOSTIC_DONE_1").is_file()
        assert (log_dir / "ENDED").is_file()


def test_generated_worker_failure_sidecar_does_not_mark_failed_command_done(
    tmp_path: Path,
) -> None:
    sidecar = _ray_template_block(
        "# RAY_WORKER_FAILURE_SIDECAR_START\n",
        "# RAY_WORKER_FAILURE_SIDECAR_END\n",
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "DRIVER_FAILED").touch()
    invocation = tmp_path / "invocation"
    failure_command = tmp_path / "failure-command.sh"
    failure_command.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$FAILURE_DIAGNOSTIC_NODE_INDEX\" > {shlex.quote(str(invocation))}\n"
        "exit 23\n"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    f"LOG_DIR={shlex.quote(str(log_dir))}",
                    f"FAILURE_COMMAND_FILE={shlex.quote(str(failure_command))}",
                    "SLURM_PROCID=2",
                    sidecar,
                    "wait",
                )
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert invocation.read_text() == "3\n"
    assert not (log_dir / "FAILURE_DIAGNOSTIC_DONE_3").exists()
    assert not (log_dir / "ENDED").exists()


def test_generated_head_failure_hook_bounds_slow_collection_and_merge(
    tmp_path: Path,
) -> None:
    hook = _ray_template_block("# RAY_FAILURE_HOOK_START\n", "# RAY_FAILURE_HOOK_END\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    invocations = tmp_path / "invocations"
    failure_command = tmp_path / "slow-failure-command.sh"
    failure_command.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"${{FAILURE_DIAGNOSTIC_MERGE:-0}}\" >> {shlex.quote(str(invocations))}\n"
        "command_deadline=$((SECONDS + 3))\n"
        "while (( SECONDS < command_deadline )); do :; done\n"
        f"touch {shlex.quote(str(log_dir / 'FAILURE_DIAGNOSTIC_DONE_1'))}\n"
    )
    started = time.monotonic()
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    f"LOG_DIR={shlex.quote(str(log_dir))}",
                    f"FAILURE_COMMAND_FILE={shlex.quote(str(failure_command))}",
                    "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=2",
                    "SLURM_JOB_NUM_NODES=2",
                    "exit_code=17",
                    hook,
                    'exit "$exit_code"',
                )
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 17, result.stderr
    assert elapsed < 4
    assert invocations.read_text().splitlines() == ["0", "1"]
    assert not (log_dir / "FAILURE_DIAGNOSTIC_DONE_0").exists()
    assert (log_dir / "ENDED").is_file()


def test_generated_head_failure_hook_merges_while_worker_diagnostic_hangs(
    tmp_path: Path,
) -> None:
    hook = _ray_template_block("# RAY_FAILURE_HOOK_START\n", "# RAY_FAILURE_HOOK_END\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    invocations = tmp_path / "invocations"
    summary = tmp_path / "summary.json"
    failure_command = tmp_path / "failure-command.sh"
    failure_command.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"${{FAILURE_DIAGNOSTIC_MERGE:-0}}\" >> {shlex.quote(str(invocations))}\n"
        'if [[ "${FAILURE_DIAGNOSTIC_MERGE:-0}" == "1" ]]; then\n'
        f"  printf '%s\\n' '{{\"missing_nodes\":[1]}}' > {shlex.quote(str(summary))}\n"
        "fi\n"
    )
    started = time.monotonic()
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    f"LOG_DIR={shlex.quote(str(log_dir))}",
                    f"FAILURE_COMMAND_FILE={shlex.quote(str(failure_command))}",
                    "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=2",
                    "SLURM_JOB_NUM_NODES=2",
                    "exit_code=19",
                    "(worker_deadline=$((SECONDS + 4)); "
                    "while (( SECONDS < worker_deadline )); do :; done; "
                    f"touch {shlex.quote(str(log_dir / 'FAILURE_DIAGNOSTIC_DONE_1'))}) &",
                    "late_worker_pid=$!",
                    hook,
                    'kill "$late_worker_pid" 2>/dev/null || true',
                    'wait "$late_worker_pid" 2>/dev/null || true',
                )
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    assert elapsed < 3.5
    assert invocations.read_text().splitlines() == ["0", "1"]
    assert json.loads(summary.read_text()) == {"missing_nodes": [1]}
    assert (log_dir / "ENDED").is_file()


def test_generated_head_failure_hook_rejects_invalid_timeout_and_cleans_up(
    tmp_path: Path,
) -> None:
    hook = _ray_template_block("# RAY_FAILURE_HOOK_START\n", "# RAY_FAILURE_HOOK_END\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    invocation = tmp_path / "invocation"
    failure_command = tmp_path / "failure-command.sh"
    failure_command.write_text(
        f"touch {shlex.quote(str(invocation))}\n"
        f"touch {shlex.quote(str(log_dir / 'FAILURE_DIAGNOSTIC_DONE_1'))}\n"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    f"LOG_DIR={shlex.quote(str(log_dir))}",
                    f"FAILURE_COMMAND_FILE={shlex.quote(str(failure_command))}",
                    "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=invalid",
                    "SLURM_JOB_NUM_NODES=2",
                    "exit_code=29",
                    hook,
                    'exit "$exit_code"',
                )
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 29
    assert not invocation.exists()
    assert (log_dir / "ENDED").is_file()


def test_generated_head_failure_hook_rejects_timeout_above_sixty_seconds(
    tmp_path: Path,
) -> None:
    hook = _ray_template_block("# RAY_FAILURE_HOOK_START\n", "# RAY_FAILURE_HOOK_END\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    invocation = tmp_path / "invocation"
    failure_command = tmp_path / "failure-command.sh"
    failure_command.write_text(f"touch {shlex.quote(str(invocation))}\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    f"LOG_DIR={shlex.quote(str(log_dir))}",
                    f"FAILURE_COMMAND_FILE={shlex.quote(str(failure_command))}",
                    "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=61",
                    "SLURM_JOB_NUM_NODES=1",
                    "exit_code=37",
                    hook,
                    'exit "$exit_code"',
                )
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 37
    assert not invocation.exists()
    assert (log_dir / "ENDED").is_file()


def test_generated_head_failure_hook_cleans_up_after_polling_command_failure(
    tmp_path: Path,
) -> None:
    hook = _ray_template_block("# RAY_FAILURE_HOOK_START\n", "# RAY_FAILURE_HOOK_END\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    invocations = tmp_path / "invocations"
    failure_command = tmp_path / "failure-command.sh"
    failure_command.write_text(
        f"printf '%s\\n' \"${{FAILURE_DIAGNOSTIC_MERGE:-0}}\" >> {shlex.quote(str(invocations))}\n"
    )
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    for command in ("find", "sleep"):
        executable = mock_bin / command
        executable.write_text("#!/bin/bash\nexit 7\n")
        executable.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{mock_bin}:{env['PATH']}"
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -euo pipefail",
                    f"LOG_DIR={shlex.quote(str(log_dir))}",
                    f"FAILURE_COMMAND_FILE={shlex.quote(str(failure_command))}",
                    "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=4",
                    "SLURM_JOB_NUM_NODES=2",
                    "exit_code=31",
                    hook,
                    'exit "$exit_code"',
                )
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 31, result.stderr
    assert invocations.read_text().splitlines() == ["0", "1"]
    assert (log_dir / "ENDED").is_file()


def test_submitter_wires_sanitized_triton_failure_command() -> None:
    source = SUBMITTER.read_text()
    assert "collect_triton_cache_diagnostics.py" in source
    assert "--from-slurm-env" in source
    assert "printf -v FAILURE_COMMAND" in source
    assert "exec python3 %q" in source
    assert source.count("-u FAILURE_COMMAND") == 2
    assert source.count("-u FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS") == 2
    assert source.count("-u CUTEDSL_BENCHMARK_RESULT_ROOT") == 2
    assert source.count('"FAILURE_COMMAND=${FAILURE_COMMAND}"') == 2
    assert (
        source.count(
            '"FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS=${FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS}"'
        )
        == 2
    )
    assert source.count('"CUTEDSL_BENCHMARK_RESULT_ROOT=${RESULT_ROOT}"') == 2
    assert source.count("-u CUTEDSL_SHARED_HF_HOME") == 2
    assert source.count('"CUTEDSL_SHARED_HF_HOME=${CUTEDSL_SHARED_HF_HOME}"') == 2
    assert (
        'RAY_MOUNTS+=",${CUTEDSL_SHARED_HF_HOME}:${CUTEDSL_SHARED_HF_HOME}"' in source
    )


def test_matrix_result_root_matches_failure_diagnostic_root() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert (
        'RESULT_ROOT="${CUTEDSL_BENCHMARK_RESULT_ROOT:-${EXPERIMENT_DIR}/results}"'
        in source
    )
    assert 'readonly RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"' in source


def test_source_pin_is_portable_across_isolated_feature_branches() -> None:
    source = PROFILE_LOADER.read_text()
    assert 'CUTEDSL_REQUIRED_GIT_BRANCH="sna/nemo-2606-cutedsl-20260710"' not in source
    command = f"""
set -euo pipefail
source {PROFILE_LOADER!s}
capture_cutedsl_submission_source {PROJECT_ROOT!s}
validate_cutedsl_runtime_source "$CUTEDSL_SUBMISSION_GIT_BRANCH" "$CUTEDSL_SUBMISSION_GIT_SHA"
printf '%s@%s\n' "$CUTEDSL_SUBMISSION_GIT_BRANCH" "$CUTEDSL_SUBMISSION_GIT_SHA"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    expected_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
    ).strip()
    expected_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    assert result.stdout.strip() == f"{expected_branch}@{expected_sha}"


def test_submitter_marks_full_cg_cutedsl_off_as_not_applicable() -> None:
    source = SUBMITTER.read_text()
    assert 'CONTEXTS="${NEMO2606_FACTORIAL_CONTEXTS:-g0a0,g0a1}"' in source
    assert 'REPLICATES="${NEMO2606_FACTORIAL_REPLICATES:-3}"' in source
    assert 'WARMUP_UPDATES="${NEMO2606_FACTORIAL_WARMUP_UPDATES:-5}"' in source
    assert 'MEASURED_UPDATES="${NEMO2606_FACTORIAL_MEASURED_UPDATES:-20}"' in source
    assert "((REPLICATES < 3))" in source
    assert 'timing_order="on,off"' in source
    assert 'timing_order="off,on"' in source
    assert 'if [[ "${full_cg_enabled}" == "1" ]]; then' in source
    assert 'timing_order="on"' in source
    assert '"NEMO2606_FULL_CG_ENABLED=${full_cg_enabled}"' in source
    assert '"NEMO2606_A2A_ENABLED=${a2a_enabled}"' in source
    assert '"CUTEDSL_BENCHMARK_ORDER=${timing_order}"' in source
    assert 'if [[ "${TEST_ONLY}" == "0" && "${needs_a2a}" == "1" ]]' in source
    assert 'if [[ "${TEST_ONLY}" == "0" && "${needs_full_cg}" == "1" ]]' in source
    assert source.index('needs_full_cg="0"') < source.index("job_id=$(sbatch")


def test_submitter_test_only_exports_runnable_default_contexts(tmp_path: Path) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    mock_sbatch = mock_bin / "sbatch"
    mock_sbatch.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

payload_arg = next(arg for arg in sys.argv[1:] if arg.startswith("--export-file="))
payload = {}
for entry in Path(payload_arg.split("=", 1)[1]).read_bytes().split(b"\\0"):
    if entry:
        key, value = entry.decode().split("=", 1)
        payload[key] = value
record = {
    "argv": ["--export-file=<payload>" if arg == payload_arg else arg for arg in sys.argv[1:]],
    "context": payload["NEMO2606_FACTORIAL_CONTEXT"],
    "full_cg": payload["NEMO2606_FULL_CG_ENABLED"],
    "a2a": payload["NEMO2606_A2A_ENABLED"],
    "replicate": payload["CUTEDSL_BENCHMARK_REPLICATE"],
    "order": payload["CUTEDSL_BENCHMARK_ORDER"],
    "profile": payload["CUTEDSL_BENCHMARK_PROFILE"],
    "warmup_updates": payload["CUTEDSL_BENCHMARK_WARMUP_UPDATES"],
    "measured_updates": payload["CUTEDSL_BENCHMARK_MEASURED_UPDATES"],
    "existing_ray": payload["CUTEDSL_BENCHMARK_EXISTING_RAY"],
    "nodes": payload["CUTEDSL_BENCHMARK_NUM_NODES"],
    "segment_size": payload["CUTEDSL_BENCHMARK_SEGMENT_SIZE"],
    "gpus_per_node": payload["GPUS_PER_NODE"],
    "command": payload["COMMAND"],
    "container": payload["CONTAINER"],
    "mounts": payload["MOUNTS"],
    "setup_command": payload["SETUP_COMMAND"],
    "failure_command": payload["FAILURE_COMMAND"],
    "failure_diagnostic_timeout_seconds": payload["FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS"],
    "result_root": payload["CUTEDSL_BENCHMARK_RESULT_ROOT"],
    "shared_hf_home": payload["CUTEDSL_SHARED_HF_HOME"],
}
with Path(os.environ["MOCK_SBATCH_CALLS"]).open("a") as output:
    output.write(json.dumps(record) + "\\n")
print(f"mock-{record['context']}-{record['replicate']}")
"""
    )
    mock_sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
            "CUTEDSL_SHARED_HF_HOME": "/stale/hf_home",
            "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS": "600",
        }
    )
    result = subprocess.run(
        ["bash", str(SUBMITTER), "--test-only"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 6
    assert [call["context"] for call in calls] == [
        "g0a0",
        "g0a1",
        "g0a1",
        "g0a0",
        "g0a0",
        "g0a1",
    ]
    for context in ("g0a0", "g0a1"):
        context_calls = [call for call in calls if call["context"] == context]
        assert [call["order"] for call in context_calls] == [
            "on,off",
            "off,on",
            "on,off",
        ]
        assert [call["replicate"] for call in context_calls] == ["0", "1", "2"]
    for call in calls:
        assert call["existing_ray"] == "1"
        assert call["nodes"] == "2"
        assert call["segment_size"] == "2"
        assert call["gpus_per_node"] == "4"
        assert call["command"].endswith("/run_cutedsl_matrix.sbatch")
        assert call["container"].endswith(".sqsh")
        assert call["mounts"].startswith(f"{PROJECT_ROOT}:{PROJECT_ROOT},")
        image_mount = f"{call['container']}:{call['container']}"
        assert call["mounts"].endswith(image_mount)
        assert ".shared_fs_canary" in call["setup_command"]
        assert "git -C" in call["setup_command"]
        assert call["result_root"] == str(EXPERIMENT_DIR / "results")
        assert call["shared_hf_home"] == (
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"
        )
        assert "exec python3" in call["failure_command"]
        assert "collect_triton_cache_diagnostics.py" in call["failure_command"]
        assert "--from-slurm-env" in call["failure_command"]
        assert ".venv" not in call["failure_command"]
        assert call["failure_diagnostic_timeout_seconds"] == "60"
        syntax = subprocess.run(
            ["bash", "-n", "-c", call["failure_command"]],
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr
        assert "--nodes=2" in call["argv"]
        assert "--segment=2" in call["argv"]
        assert "--segment=1" not in call["argv"]
        assert "--test-only" in call["argv"]
        assert call["argv"][-1] == str(PROJECT_ROOT / "ray.sub")

    calls_path.unlink()
    env["NEMO2606_FACTORIAL_CONTEXTS"] = "g0a0,g1a0,g0a1,g1a1"
    all_contexts = subprocess.run(
        ["bash", str(SUBMITTER), "--test-only"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert all_contexts.returncode == 0, all_contexts.stderr
    all_calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(all_calls) == 12
    for context in ("g0a0", "g1a0", "g0a1", "g1a1"):
        context_calls = [call for call in all_calls if call["context"] == context]
        expected_orders = (
            ["on", "on", "on"]
            if context.startswith("g1")
            else ["on,off", "off,on", "on,off"]
        )
        assert [call["order"] for call in context_calls] == expected_orders
        assert [call["replicate"] for call in context_calls] == ["0", "1", "2"]


def test_official_submitter_exports_matched_4n4g_performance_jobs(
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    mock_sbatch = mock_bin / "sbatch"
    mock_sbatch.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

payload_arg = next(arg for arg in sys.argv[1:] if arg.startswith("--export-file="))
payload = {}
for entry in Path(payload_arg.split("=", 1)[1]).read_bytes().split(b"\\0"):
    if entry:
        key, value = entry.decode().split("=", 1)
        payload[key] = value
record = {
    "argv": ["--export-file=<payload>" if arg == payload_arg else arg for arg in sys.argv[1:]],
    "context": payload["NEMO2606_FACTORIAL_CONTEXT"],
    "recipe": payload["CUTEDSL_BENCHMARK_RECIPE"],
    "nodes": payload["CUTEDSL_BENCHMARK_NUM_NODES"],
    "gpus_per_node": payload["CUTEDSL_BENCHMARK_GPUS_PER_NODE"],
    "segment_size": payload["CUTEDSL_BENCHMARK_SEGMENT_SIZE"],
    "train_global_batch_size": payload["CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE"],
    "expert_model_parallel_size": payload["CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE"],
    "replicate": payload["CUTEDSL_BENCHMARK_REPLICATE"],
    "order": payload["CUTEDSL_BENCHMARK_ORDER"],
    "profile": payload["CUTEDSL_BENCHMARK_PROFILE"],
    "warmup_updates": payload["CUTEDSL_BENCHMARK_WARMUP_UPDATES"],
    "measured_updates": payload["CUTEDSL_BENCHMARK_MEASURED_UPDATES"],
    "full_cg": payload["NEMO2606_FULL_CG_ENABLED"],
    "a2a": payload["NEMO2606_A2A_ENABLED"],
    "existing_ray": payload["CUTEDSL_BENCHMARK_EXISTING_RAY"],
    "shared_hf_home": payload["CUTEDSL_SHARED_HF_HOME"],
}
with Path(os.environ["MOCK_SBATCH_CALLS"]).open("a") as output:
    output.write(json.dumps(record) + "\\n")
print(f"mock-{record['context']}-{record['replicate']}")
"""
    )
    mock_sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
        }
    )
    result = subprocess.run(
        ["bash", str(OFFICIAL_SUBMITTER), "--test-only"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 3
    assert [call["context"] for call in calls] == ["g0a0", "g0a0", "g0a0"]
    assert [call["replicate"] for call in calls] == ["0", "1", "2"]
    assert [call["order"] for call in calls] == ["on,off", "off,on", "on,off"]
    assert [call["profile"] for call in calls] == ["1", "0", "0"]
    for call in calls:
        assert call["recipe"].endswith(
            "grpo-qwen3-30ba3b-4n4g-megatron-mxfp8-cutedsl.yaml"
        )
        assert call["nodes"] == "4"
        assert call["gpus_per_node"] == "4"
        assert call["segment_size"] == "4"
        assert call["train_global_batch_size"] == "2048"
        assert call["expert_model_parallel_size"] == "16"
        assert call["warmup_updates"] == "5"
        assert call["measured_updates"] == "20"
        assert call["full_cg"] == "0"
        assert call["a2a"] == "0"
        assert call["existing_ray"] == "1"
        assert call["shared_hf_home"] == (
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home"
        )
        assert "--nodes=4" in call["argv"]
        assert "--segment=4" in call["argv"]
        assert "--segment=1" not in call["argv"]
        assert "--exclusive" in call["argv"]
        assert "--time=05:00:00" in call["argv"]
        assert "--account=coreai_dlalgo_llm" in call["argv"]
        assert "--partition=batch" in call["argv"]
        assert "--comment=metrics" in call["argv"]
        assert "--test-only" in call["argv"]


@pytest.mark.parametrize(
    (
        "submitter",
        "expected_nodes",
        "expected_segment_size",
        "expected_train_global_batch_size",
        "expected_expert_model_parallel_size",
    ),
    (
        (SUBMITTER, "2", "2", "16", "8"),
        (OFFICIAL_SUBMITTER, "4", "4", "2048", "16"),
    ),
)
def test_functional_submitter_exports_one_fail_closed_job(
    tmp_path: Path,
    submitter: Path,
    expected_nodes: str,
    expected_segment_size: str,
    expected_train_global_batch_size: str,
    expected_expert_model_parallel_size: str,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    mock_sbatch = mock_bin / "sbatch"
    mock_sbatch.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

payload_arg = next(arg for arg in sys.argv[1:] if arg.startswith("--export-file="))
payload = {}
for entry in Path(payload_arg.split("=", 1)[1]).read_bytes().split(b"\\0"):
    if entry:
        key, value = entry.decode().split("=", 1)
        payload[key] = value
record = {
    "argv": ["--export-file=<payload>" if arg == payload_arg else arg for arg in sys.argv[1:]],
    "functional_gate": payload.get("NEMO2606_FUNCTIONAL_GATE"),
    "functional_updates": payload.get("NEMO2606_FUNCTIONAL_UPDATES"),
    "context": payload.get("NEMO2606_FACTORIAL_CONTEXT"),
    "order": payload.get("CUTEDSL_BENCHMARK_ORDER"),
    "profile": payload.get("CUTEDSL_BENCHMARK_PROFILE"),
    "existing_ray": payload.get("CUTEDSL_BENCHMARK_EXISTING_RAY"),
    "nodes": payload.get("CUTEDSL_BENCHMARK_NUM_NODES"),
    "segment_size": payload.get("CUTEDSL_BENCHMARK_SEGMENT_SIZE"),
    "gpus_per_node": payload.get("CUTEDSL_BENCHMARK_GPUS_PER_NODE"),
    "train_global_batch_size": payload.get("CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE"),
    "expert_model_parallel_size": payload.get("CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE"),
    "full_cg": payload.get("NEMO2606_FULL_CG_ENABLED"),
    "a2a": payload.get("NEMO2606_A2A_ENABLED"),
    "failure_diagnostic_timeout_seconds": payload.get("FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS"),
    "shared_hf_home": payload.get("CUTEDSL_SHARED_HF_HOME"),
}
with Path(os.environ["MOCK_SBATCH_CALLS"]).open("a") as output:
    output.write(json.dumps(record) + "\\n")
print("mock-functional")
"""
    )
    mock_sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
            "FAILURE_DIAGNOSTIC_TIMEOUT_SECONDS": "600",
            "NEMO2606_FUNCTIONAL_GATE": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(submitter), "--test-only"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 1
    call = calls[0]
    assert call["functional_gate"] == "1"
    assert call["functional_updates"] == "3"
    assert call["context"] == "g0a0"
    assert call["order"] == "on"
    assert call["profile"] == "0"
    assert call["existing_ray"] == "1"
    assert call["nodes"] == expected_nodes
    assert call["segment_size"] == expected_segment_size
    assert call["gpus_per_node"] == "4"
    assert call["train_global_batch_size"] == expected_train_global_batch_size
    assert call["expert_model_parallel_size"] == expected_expert_model_parallel_size
    assert call["full_cg"] == "0"
    assert call["a2a"] == "0"
    assert call["failure_diagnostic_timeout_seconds"] == "60"
    assert call["shared_hf_home"] == ("/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home")
    assert f"--nodes={expected_nodes}" in call["argv"]
    assert f"--segment={expected_segment_size}" in call["argv"]
    assert "--test-only" in call["argv"]


def test_matrix_payload_reuses_collectors_in_existing_ray_mode() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert 'EXISTING_RAY="${CUTEDSL_BENCHMARK_EXISTING_RAY:-0}"' in source
    assert 'if [[ "${EXISTING_RAY}" == "1" ]]; then' in source
    assert "SRUN=()" in source
    assert 'export NEMO_RL_VENV_DIR="${NODE_LOCAL_WORKER_VENV_ROOT}"' in source
    assert 'RAY_LOG_ATTEMPT_ID="${SLURM_JOB_ID}-${SLURM_RESTART_COUNT}"' in source
    assert 'RAY_CLUSTER_LOG_DIR="${BASE_LOG_DIR:' in source
    assert '${RAY_LOG_ATTEMPT_ID}-logs/ray"' in source
    assert "cluster.num_nodes=${BENCHMARK_NUM_NODES}" in source
    assert "cluster.gpus_per_node=${BENCHMARK_GPUS_PER_NODE}" in source
    assert (
        "policy.megatron_cfg.expert_model_parallel_size=${EXPERT_MODEL_PARALLEL_SIZE}"
        in source
    )
    assert "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}" in source
    assert (
        'TRAIN_GLOBAL_BATCH_SIZE="${CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE:'
        in source
    )
    assert (
        'EXPERT_MODEL_PARALLEL_SIZE="${CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE:'
        in source
    )
    assert 'TRAINING_GPU_COUNT = int(os.environ["TRAINING_GPU_COUNT"])' in source
    for metric in (
        "timing/train/total_step_time",
        "timing/train/generation",
        "timing/train/get_logprobs",
        "timing/train/policy_training",
        "timing/train/prepare_for_generation/transfer_and_update_weights",
        "performance/tokens_per_sec_per_gpu",
        "performance/generation_tokens_per_sec_per_gpu",
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
        "performance/policy_training_tokens_per_sec_per_gpu",
    ):
        assert metric in source


def test_matrix_fails_closed_without_noneditable_submission_contract() -> None:
    source = MATRIX_PAYLOAD.read_text()
    contract = 'if [[ "${UV_NO_EDITABLE-}" != "1" ]]; then'
    runtime_bootstrap = "cutedsl_write_event runtime_bootstrap start"

    assert contract in source
    assert "UV_NO_EDITABLE=1 is required for isolated local-package builds." in source
    assert source.index(contract) < source.index(runtime_bootstrap)
    for value in (None, "0"):
        env = os.environ.copy()
        if value is None:
            env.pop("UV_NO_EDITABLE", None)
        else:
            env["UV_NO_EDITABLE"] = value
        result = subprocess.run(
            ["bash", str(MATRIX_PAYLOAD)],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert (
            result.stderr
            == "[ERROR] UV_NO_EDITABLE=1 is required for isolated local-package builds.\n"
        )


def test_direct_pyxis_preserves_noneditable_contract_without_host_paths() -> None:
    source = MATRIX_PAYLOAD.read_text()
    match = re.search(r"^\s*--container-env=([^\s]+)$", source, re.MULTILINE)

    assert match is not None
    forwarded_names = match.group(1).split(",")
    assert forwarded_names == [
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "NVTE_CUDA_ARCHS",
        "UV_NO_EDITABLE",
    ]
    assert "PATH" not in forwarded_names
    assert "PYTHONPATH" not in forwarded_names

    image_env = {
        "UV_NO_EDITABLE": "0",
        "PATH": "/image/bin",
        "PYTHONPATH": "/image/python",
    }
    host_env = {"UV_NO_EDITABLE": "1"}
    resolved = image_env.copy()
    for name in forwarded_names:
        if name in host_env:
            resolved[name] = host_env[name]

    assert resolved == {
        "UV_NO_EDITABLE": "1",
        "PATH": "/image/bin",
        "PYTHONPATH": "/image/python",
    }


def test_every_direct_srun_reprefixes_container_runtime_paths(
    tmp_path: Path,
) -> None:
    source = MATRIX_PAYLOAD.read_text()
    payloads = re.findall(
        r'^\s*"\$\{SRUN\[@\]\}" bash -s <<\'BASH\'\n(.*?)^BASH$',
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert len(payloads) == 4

    image_bin = tmp_path / "image-bin"
    image_python = tmp_path / "image-python"
    image_bin.mkdir()
    image_python.mkdir()
    (image_bin / "nsys").write_text("#!/bin/sh\nexit 0\n")
    (image_bin / "nsys").chmod(0o755)
    (image_bin / "python3").symlink_to(sys.executable)

    for index, payload in enumerate(payloads):
        preamble = payload.split("printf '[INFO] Container temporary directory:", 1)[0]
        case_root = tmp_path / f"srun-{index}"
        runtime_root = case_root / "runtime"
        runtime_bin = runtime_root / "runtime-bin"
        overlay = runtime_root / "mcore-overlay"
        venv = runtime_root / "venv"
        repository = case_root / "repo"
        repository.mkdir(parents=True)

        probe = [
            'printf "PATH=%s\\nPYTHONPATH=%s\\nUV_NO_EDITABLE=%s\\n" "${PATH}" "${PYTHONPATH}" "${UV_NO_EDITABLE}"',
            "command -v nsys",
        ]
        if index == 0:
            probe.append('[[ ! -e "${MCORE_OVERLAY_ROOT}" ]]')
        else:
            runtime_bin.mkdir(parents=True)
            (runtime_bin / "python3-config").write_text("#!/bin/sh\nexit 0\n")
            (runtime_bin / "python3-config").chmod(0o755)
            overlay.mkdir(parents=True)
            (overlay / "mcore_overlay_sentinel.py").write_text(
                'ORIGIN = "run-scoped-overlay"\n'
            )
            probe.extend(
                [
                    "command -v python3-config",
                    "python3 -c 'import mcore_overlay_sentinel; print(mcore_overlay_sentinel.ORIGIN)'",
                ]
            )

        env = os.environ.copy()
        env.update(
            {
                "CONTAINER_REPO_ROOT": str(repository),
                "CONTAINER_RUNTIME_DIR": str(runtime_root),
                "MCORE_OVERLAY_ROOT": str(overlay),
                "PATH": str(image_bin),
                "PYTHONPATH": str(image_python),
                "RUNTIME_TOOL_BIN": str(runtime_bin),
                "UV_NO_EDITABLE": "1",
                "UV_PROJECT_ENVIRONMENT": str(venv),
            }
        )
        result = subprocess.run(
            ["/bin/bash", "-c", preamble + "\n" + "\n".join(probe)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (index, result.stderr)
        lines = result.stdout.splitlines()
        assert lines[:4] == [
            f"PATH={runtime_bin}:{venv / 'bin'}:{image_bin}",
            f"PYTHONPATH={overlay}:{repository}:{image_python}",
            "UV_NO_EDITABLE=1",
            str(image_bin / "nsys"),
        ]
        if index > 0:
            assert lines[4:] == [
                str(runtime_bin / "python3-config"),
                "run-scoped-overlay",
            ]

        already_prefixed_env = env.copy()
        already_prefixed_env["PATH"] = f"{runtime_bin}:{venv / 'bin'}:{image_bin}"
        already_prefixed_env["PYTHONPATH"] = f"{overlay}:{repository}:{image_python}"
        existing_ray_result = subprocess.run(
            ["/bin/bash", "-c", preamble + "\n" + "\n".join(probe)],
            env=already_prefixed_env,
            capture_output=True,
            text=True,
        )
        assert existing_ray_result.returncode == 0, (
            index,
            existing_ray_result.stderr,
        )
        assert existing_ray_result.stdout.splitlines() == lines


def test_bootstrap_refreshes_cached_python_after_locked_sync(tmp_path: Path) -> None:
    source = MATRIX_PAYLOAD.read_text()
    refresh_start = "# CUTEDSL_RUNTIME_PYTHON_REFRESH_START"
    refresh_end = "# CUTEDSL_RUNTIME_PYTHON_REFRESH_END"

    assert source.count(refresh_start) == 1
    assert source.count(refresh_end) == 1
    refresh = source.split(f"{refresh_start}\n", 1)[1].split(f"{refresh_end}\n", 1)[0]
    locked_sync = (
        '"${UV_BIN}" sync --locked --extra mcore --group test --group dev '
        '--python "${UV_PYTHON_VERSION}"'
    )
    assert source.index(f"{locked_sync}\n{refresh_start}") >= 0
    assert refresh.splitlines()[0] == "hash -r"
    assert 'resolved_runtime_python="$(command -v python3)"' in refresh
    assert '"${UV_PROJECT_ENVIRONMENT}/bin/python3"' in refresh

    image_bin = tmp_path / "image-bin"
    venv_bin = tmp_path / "venv" / "bin"
    image_bin.mkdir()
    image_python = image_bin / "python3"
    image_python.write_text("#!/bin/sh\nprintf 'image-python\\n'\n")
    image_python.chmod(0o755)

    script = f"""\
set -euo pipefail
python3 > "${{UV_PROJECT_ENVIRONMENT}}.first-python"
first_python=$(cat "${{UV_PROJECT_ENVIRONMENT}}.first-python")
[[ "${{first_python}}" == "image-python" ]]
mkdir -p "${{UV_PROJECT_ENVIRONMENT}}/bin"
cat > "${{UV_PROJECT_ENVIRONMENT}}/bin/python3" <<'PYTHON'
#!/bin/sh
printf 'runtime-python\\n'
PYTHON
chmod +x "${{UV_PROJECT_ENVIRONMENT}}/bin/python3"
{refresh}
printf '%s\\n' "$(command -v python3)"
python3
"""
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        env={
            **os.environ,
            "PATH": f"{venv_bin}:{image_bin}:{os.environ['PATH']}",
            "UV_PROJECT_ENVIRONMENT": str(venv_bin.parent),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(venv_bin / "python3"),
        "runtime-python",
    ]


def test_matrix_builds_tracked_mcore_overlay_before_config_resolution() -> None:
    source = MATRIX_PAYLOAD.read_text()
    source_preflight = 'SOURCE_STATUS=$(git -C "${REPO_ROOT}" status'
    overlay_start = "# CUTEDSL_MCORE_OVERLAY_BOOTSTRAP_START"
    config_resolution = "from nemo_rl.algorithms.grpo import MasterConfig"
    first_grpo = '"${UV_BIN}" run --active --no-sync examples/run_grpo.py'
    required = (
        'git -C "${HOST_MCORE_SOURCE_ROOT}" archive --format=tar',
        'tar -xf "${CONTAINER_MCORE_ARCHIVE}" -C "${MCORE_OVERLAY_ROOT}"',
        'ln -s "${runtime_python_config}" "${RUNTIME_TOOL_BIN}/python3-config"',
        'make -C "${MCORE_OVERLAY_DATASETS}"',
        'sysconfig.get_config_var("EXT_SUFFIX")',
        'make -q -C "${MCORE_OVERLAY_DATASETS}"',
        "import megatron.core.datasets.helpers_cpp as helper_module",
        "Path(helper_module.__file__).resolve()",
    )

    for fragment in required:
        assert fragment in source, fragment
    assert source.count('make -C "${MCORE_OVERLAY_DATASETS}"') == 1
    assert source.index(source_preflight) < source.index(overlay_start)
    assert source.index(overlay_start) < source.index(config_resolution)
    assert source.index(overlay_start) < source.index(first_grpo)


def test_direct_pyxis_bootstrap_extracts_host_archive_without_git_metadata(
    tmp_path: Path,
) -> None:
    source = MATRIX_PAYLOAD.read_text()
    host_start = "# CUTEDSL_MCORE_HOST_ARCHIVE_START"
    host_end = "# CUTEDSL_MCORE_HOST_ARCHIVE_END"
    extract_start = "# CUTEDSL_MCORE_OVERLAY_EXTRACT_START"
    extract_end = "# CUTEDSL_MCORE_OVERLAY_EXTRACT_END"

    assert host_start in source
    assert host_end in source
    assert extract_start in source
    assert extract_end in source
    host_archive = source.split(f"{host_start}\n", 1)[1].split(f"{host_end}\n", 1)[0]
    container_extract = source.split(f"{extract_start}\n", 1)[1].split(
        f"{extract_end}\n", 1
    )[0]
    assert 'git -C "${HOST_MCORE_SOURCE_ROOT}" archive' in host_archive
    assert 'tar -xf "${CONTAINER_MCORE_ARCHIVE}"' in container_extract
    assert "git " not in container_extract

    repository = tmp_path / "git-common" / "source"
    repository.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test User"],
        check=True,
    )
    tracked_file = repository / "megatron" / "core" / "datasets" / "Makefile"
    tracked_file.parent.mkdir(parents=True)
    tracked_file.write_text("default:\n\t@true\n")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "tracked source"],
        check=True,
    )
    source_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    host_source = tmp_path / "host" / "deep" / "Megatron-LM"
    host_source.parent.mkdir(parents=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "worktree",
            "add",
            "--detach",
            "-q",
            str(host_source),
            source_sha,
        ],
        check=True,
    )
    gitdir = Path((host_source / ".git").read_text().strip().split("gitdir: ", 1)[1])
    (host_source / ".git").write_text(
        f"gitdir: {os.path.relpath(gitdir, host_source)}\n"
    )
    subprocess.run(
        ["git", "-C", str(host_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )

    host_runtime = tmp_path / "host-runtime"
    host_runtime.mkdir()
    archive_result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{host_archive}"],
        env={
            **os.environ,
            "HOST_MCORE_SOURCE_ROOT": str(host_source),
            "HOST_RUNTIME_DIR": str(host_runtime),
            "CONTAINER_RUNTIME_DIR": "/runtime",
            "MCORE_SOURCE_SHA": source_sha,
        },
        capture_output=True,
        text=True,
    )
    assert archive_result.returncode == 0, archive_result.stderr
    archive = host_runtime / "mcore-source.tar"
    assert archive.is_file()

    remapped_source = tmp_path / "container" / "Megatron-LM"
    remapped_source.parent.mkdir(parents=True)
    shutil.copytree(host_source, remapped_source)
    inaccessible_git = subprocess.run(
        ["git", "-C", str(remapped_source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    assert inaccessible_git.returncode != 0

    overlay = tmp_path / "container-runtime" / "mcore-overlay"
    extract_result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{container_extract}"],
        env={
            **os.environ,
            "CONTAINER_MCORE_ARCHIVE": str(archive),
            "MCORE_OVERLAY_ROOT": str(overlay),
            "RUNTIME_TOOL_BIN": str(tmp_path / "container-runtime" / "runtime-bin"),
        },
        capture_output=True,
        text=True,
    )
    assert extract_result.returncode == 0, extract_result.stderr
    assert (overlay / "megatron" / "core" / "datasets" / "Makefile").read_text() == (
        "default:\n\t@true\n"
    )


def test_canonical_source_is_rechecked_by_outer_shell_after_bootstrap() -> None:
    source = MATRIX_PAYLOAD.read_text()
    bootstrap_call = "\"${SRUN[@]}\" bash -s <<'BASH'"
    bootstrap_start = source.index(bootstrap_call)
    bootstrap_end = source.index("\nBASH\n", bootstrap_start) + len("\nBASH\n")
    post_start = "# CUTEDSL_MCORE_CANONICAL_POST_BOOTSTRAP_START"
    post_end = "# CUTEDSL_MCORE_CANONICAL_POST_BOOTSTRAP_END"

    assert post_start in source
    assert post_end in source
    post_check_start = source.index(post_start)
    post_check_end = source.index(post_end)
    post_check = source[post_check_start:post_check_end]
    assert bootstrap_end <= post_check_start
    assert 'reject_canonical_mcore_helpers "${HOST_MCORE_SOURCE_DATASETS}"' in (
        post_check
    )
    assert 'git -C "${canonical_root}" status --porcelain' in post_check
    assert '"${REPO_ROOT}"' in post_check
    assert '"${HOST_MCORE_BRIDGE_ROOT}"' in post_check
    assert '"${HOST_MCORE_SOURCE_ROOT}"' in post_check
    assert post_check_end < source.index(
        "from nemo_rl.algorithms.grpo import MasterConfig"
    )
    assert post_check_end < source.index("cutedsl_write_event runtime_bootstrap pass")


def test_matrix_never_builds_mcore_helpers_in_canonical_checkout(
    tmp_path: Path,
) -> None:
    source = MATRIX_PAYLOAD.read_text()
    canonical_rejection = 'reject_canonical_mcore_helpers "${MCORE_SOURCE_DATASETS}"'
    overlay_build = 'make -C "${MCORE_OVERLAY_DATASETS}"'

    assert source.count(canonical_rejection) == 2
    assert "-name 'helpers_cpp*'" in source
    assert overlay_build in source
    assert 'make -C "${MCORE_SOURCE_DATASETS}"' not in source
    assert source.index(canonical_rejection) < source.index(overlay_build)
    assert source.rindex(canonical_rejection) > source.index(overlay_build)

    function_source = (
        "reject_canonical_mcore_helpers() {"
        + source.split("reject_canonical_mcore_helpers() {", 1)[1].split("\n}\n", 1)[0]
        + "\n}"
    )
    canonical_datasets = tmp_path / "canonical-mcore-datasets"
    canonical_datasets.mkdir()
    artifact = canonical_datasets / "helpers_cpp.cpython-313-aarch64-linux-gnu.so"
    artifact.write_bytes(b"ignored ABI artifact")
    env = os.environ.copy()
    env["CANONICAL_DATASETS"] = str(canonical_datasets)
    result = subprocess.run(
        [
            "bash",
            "-c",
            function_source
            + '\nreject_canonical_mcore_helpers "${CANONICAL_DATASETS}"',
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "generated helpers_cpp artifact" in result.stderr


def test_manifest_records_mcore_runtime_isolation() -> None:
    source = MATRIX_PAYLOAD.read_text()
    expected_fields = {
        "source_sha",
        "overlay_mode",
        "uv_no_editable",
        "python_extension_suffix",
        "helper_relative_path",
        "helper_sha256",
        "helper_origin_scope",
        "python3_config_source",
        "make_query_up_to_date",
        "canonical_helper_artifact_count_before",
        "canonical_helper_artifact_count_after",
    }
    manifest_fields = re.search(
        r"MCORE_RUNTIME_FIELDS = \{(?P<fields>.*?)\n\}", source, re.DOTALL
    )

    assert manifest_fields is not None
    assert set(re.findall(r'"([a-z0-9_]+)"', manifest_fields.group("fields"))) == (
        expected_fields
    )
    assert 're.fullmatch(r"[0-9a-f]{40}", mcore_runtime["source_sha"])' in source
    assert 're.fullmatch(r"[0-9a-f]{64}", mcore_runtime["helper_sha256"])' in source
    assert '"mcore_runtime": mcore_runtime' in source


def test_mcore_overlay_is_under_existing_exit_cleanup_root() -> None:
    source = MATRIX_PAYLOAD.read_text()
    overlay_assignment = 'MCORE_OVERLAY_ROOT="${CONTAINER_RUNTIME_DIR}/mcore-overlay"'

    assert overlay_assignment in source
    assert 'RUNTIME_TOOL_BIN="${CONTAINER_RUNTIME_DIR}/runtime-bin"' in source
    assert 'rm -rf --one-file-system -- "${HOST_RUNTIME_DIR}"' in source
    assert source.index(overlay_assignment) < source.index(
        "# CUTEDSL_MCORE_OVERLAY_BOOTSTRAP_START"
    )


def test_existing_ray_uses_job_scoped_node_local_triton_cache() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert (
        'NODE_LOCAL_RUNTIME_ROOT="/tmp/${USER}/nemo2606-factorial/${RUN_ID}"' in source
    )
    assert (
        'NODE_LOCAL_WORKER_VENV_ROOT="${NODE_LOCAL_RUNTIME_ROOT}/worker_venvs"'
        in source
    )
    assert 'TRITON_CACHE_DIR="${NODE_LOCAL_RUNTIME_ROOT}/triton_cache"' in source
    assert 'NEMO2606_TRITON_CACHE_SCOPE="job_node_local"' in source
    assert '"triton_cache_scope": os.environ["NEMO2606_TRITON_CACHE_SCOPE"]' in source


def test_non_existing_ray_retains_run_local_container_cache() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert 'TRITON_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/triton_cache"' in source
    assert 'NEMO2606_TRITON_CACHE_SCOPE="run_local_container"' in source


def test_existing_ray_triton_cache_is_not_under_shared_roots() -> None:
    source = MATRIX_PAYLOAD.read_text()
    runtime_block = source.split('export NVTE_CUDA_ARCHS="100"', 1)[1]
    existing_ray = runtime_block.split('if [[ "${EXISTING_RAY}" == "1" ]]', 1)[1].split(
        "else", 1
    )[0]
    assert 'TRITON_CACHE_DIR="${NODE_LOCAL_RUNTIME_ROOT}/triton_cache"' in existing_ray
    triton_cache_assignment = existing_ray.split("TRITON_CACHE_DIR=", 1)[
        1
    ].splitlines()[0]
    assert "CONTAINER_RUNTIME_DIR" not in triton_cache_assignment
    assert "RESULT_DIR" not in triton_cache_assignment
    assert "MEGATRON_CHECKPOINT_ROOT" not in triton_cache_assignment


def test_matrix_payload_uses_shared_megatron_conversion_cache() -> None:
    source = MATRIX_PAYLOAD.read_text()
    checkpoint_root = (
        'MEGATRON_CHECKPOINT_ROOT="${CONTAINER_RUNTIME_DIR}/megatron_checkpoints"'
    )
    create_root = 'mkdir -p "${MEGATRON_CHECKPOINT_ROOT}"'
    export_root = 'export NRL_MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_ROOT}"'

    assert checkpoint_root in source
    assert create_root in source
    assert export_root in source
    assert source.index(checkpoint_root) < source.index(create_root)
    assert source.index(create_root) < source.index(export_root)


def test_functional_payload_fails_closed_before_selecting_one_arm() -> None:
    source = MATRIX_PAYLOAD.read_text()
    required = (
        'FUNCTIONAL_GATE="${NEMO2606_FUNCTIONAL_GATE:-0}"',
        'FUNCTIONAL_UPDATES="${NEMO2606_FUNCTIONAL_UPDATES:-3}"',
        '[[ "${FUNCTIONAL_GATE}" == "0" || "${FUNCTIONAL_GATE}" == "1" ]]',
        '[[ "${FUNCTIONAL_UPDATES}" == "3" ]]',
        '[[ "${TIMING_ORDER}" == "on" ]]',
        '[[ "${PROFILE_ENABLED}" == "0" ]]',
        '[[ "${FEATURE_CONTEXT}" == "g0a0" ]]',
        '[[ "${NEMO2606_FULL_CG_ENABLED}" == "0" ]]',
        '[[ "${NEMO2606_A2A_ENABLED}" == "0" ]]',
        "timing_arms=(on)",
        "WARMUP_UPDATES=0",
        "MEASURED_UPDATES=0",
        'TOTAL_UPDATES="${FUNCTIONAL_UPDATES}"',
        "else",
    )
    for fragment in required:
        assert fragment in source, fragment

    functional_branch = source.index('if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then')
    timing_branch = source.index("else", functional_branch)
    timing_order_validation = source.index(
        "CUTEDSL_BENCHMARK_ORDER must contain on and off exactly once."
    )
    assert functional_branch < timing_branch < timing_order_validation
    assert source.index("WARMUP_UPDATES < 5") > timing_branch
    assert source.index("MEASURED_UPDATES < 10") > timing_branch


def test_performance_payload_accepts_only_cutedsl_on_for_full_cg() -> None:
    source = MATRIX_PAYLOAD.read_text()
    required = (
        'if [[ "${NEMO2606_FULL_CG_ENABLED}" == "1" ]]; then',
        '[[ "${TIMING_ORDER}" == "on" ]]',
        "timing_arms=(on)",
        'expected_arms = os.environ["TIMING_ORDER"].split(",")',
        'if "off" in expected_arms:',
        '"available_arms": expected_arms',
        '"not_applicable_arms": {',
        '"off": "full-iteration CUDA Graph requires device-initiated CuTeDSL"',
        '"base_config_sha256": base_config_sha256',
        '"context_single_arm" if expected_full_cg else "context_local_cutedsl_pair"',
        '"policy.megatron_cfg.cuda_graph_impl"',
        '"policy.megatron_cfg.overlap_moe_expert_parallel_comm"',
        '"policy.megatron_cfg.high_priority_a2a_comm_stream"',
        '"policy.megatron_cfg.delay_wgrad_compute"',
        '"grpo.val_period"',
        '"grpo.val_at_start"',
        '"grpo.val_at_end"',
        '"full_cg_config_evidence": full_cg_config_evidence',
        '"cuda_graph_warmup_steps"',
        '"cuda_graph_use_single_mempool"',
        'os.environ["CUDA_GRAPH_WARMUP_STEPS"]',
    )
    for fragment in required:
        assert fragment in source, fragment


def test_base_config_identity_ignores_run_paths_and_optional_feature_keys() -> None:
    source = MATRIX_PAYLOAD.read_text()
    start_marker = "# NEMO2606_BASE_CONFIG_IDENTITY_START"
    end_marker = "# NEMO2606_BASE_CONFIG_IDENTITY_END"
    assert start_marker in source
    assert end_marker in source
    code = source.split(start_marker, 1)[1].split(end_marker, 1)[0]
    namespace: dict[str, object] = {}
    exec(
        "import hashlib\nimport json\nfrom typing import Any\n" + code,
        namespace,
    )
    digest = namespace["canonical_base_config_sha256"]
    baseline = {
        "grpo": {
            "val_period": 0,
            "val_at_start": False,
            "val_at_end": False,
        },
        "logger": {"log_dir": "/runtime/job-a/logs"},
        "checkpointing": {"checkpoint_dir": "/runtime/job-a/checkpoints"},
        "policy": {
            "train_global_batch_size": 2048,
            "megatron_cfg": {
                "env_vars": {"NVTE_CUTEDSL_FUSED_GROUPED_MLP": "1"},
                "overlap_moe_expert_parallel_comm": False,
                "high_priority_a2a_comm_stream": False,
                "delay_wgrad_compute": False,
            },
        },
    }
    full_cg = copy.deepcopy(baseline)
    full_cg["logger"]["log_dir"] = "/runtime/job-b/logs"
    full_cg["checkpointing"]["checkpoint_dir"] = "/runtime/job-b/checkpoints"
    full_cg["policy"]["megatron_cfg"].update(
        {
            "cuda_graph_impl": "full_iteration",
            "cuda_graph_warmup_steps": 3,
            "cuda_graph_use_single_mempool": True,
            "overlap_moe_expert_parallel_comm": True,
            "high_priority_a2a_comm_stream": True,
            "delay_wgrad_compute": True,
        }
    )
    assert digest(baseline) == digest(full_cg)
    changed_workload = copy.deepcopy(full_cg)
    changed_workload["policy"]["train_global_batch_size"] = 1024
    assert digest(baseline) != digest(changed_workload)
    changed_validation = copy.deepcopy(full_cg)
    changed_validation["grpo"]["val_period"] = 10
    assert digest(baseline) != digest(changed_validation)


def test_payload_rejects_feature_context_boolean_mismatch() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert (
        'case "${FEATURE_CONTEXT}:${NEMO2606_FULL_CG_ENABLED}:${NEMO2606_A2A_ENABLED}" in'
        in source
    )
    assert "g0a0:0:0|g1a0:1:0|g0a1:0:1|g1a1:1:1)" in source
    assert "Feature context does not match full-CG/A2A selectors" in source


def test_functional_payload_uses_effective_segment_and_one_arm_manifest() -> None:
    source = MATRIX_PAYLOAD.read_text()
    required = (
        'BENCHMARK_SEGMENT_SIZE="${CUTEDSL_BENCHMARK_SEGMENT_SIZE:-${CUTEDSL_SEGMENT:-1}}"',
        '"cluster.segment_size=${BENCHMARK_SEGMENT_SIZE}"',
        '"functional_gate": os.environ["FUNCTIONAL_GATE"] == "1"',
        '"performance_eligible": os.environ["FUNCTIONAL_GATE"] != "1"',
        '"segment_size": cluster_config["segment_size"]',
        '"segment": int(os.environ["BENCHMARK_SEGMENT_SIZE"])',
        'expected_arms = os.environ["TIMING_ORDER"].split(",")',
        'if "off" in expected_arms:',
        "fixed_config_evidence = {",
        '"NEMO2606_TRITON_CACHE_SCOPE": os.environ["NEMO2606_TRITON_CACHE_SCOPE"]',
        '"available_arms": expected_arms',
        '"arms": [',
    )
    for fragment in required:
        assert fragment in source, fragment
    manifest_block = source.split("manifest = {", 1)[1].split("}\n(", 1)[0]
    assert 'os.environ["CUTEDSL_SEGMENT"]' not in manifest_block


def test_functional_payload_records_three_update_component_and_runtime_evidence() -> (
    None
):
    source = MATRIX_PAYLOAD.read_text()
    required = (
        'if [[ "${FUNCTIONAL_GATE}" == "1" ]]; then',
        '"functional_gate_summary.json"',
        '"completed_updates": completed_updates',
        '"performance_eligible": False',
        '"arm": "on"',
        'TOTAL_STEP_METRIC = "timing/train/total_step_time"',
        'GENERATION_METRIC = "timing/train/generation"',
        'LOGPROB_METRIC = "timing/train/get_logprobs"',
        'POLICY_TIME_METRIC = "timing/train/policy_training"',
        'REFIT_METRIC = "timing/train/prepare_for_generation/transfer_and_update_weights"',
        'if completed_updates != int(os.environ["FUNCTIONAL_UPDATES"]):',
        'get("event") == "megatron_policy_offload_memory"',
        "phase=after_completion",
        "offload_sequence=3",
        "global_rank",
        "cgroup_memory_peak_gib",
        "cgroup_memory_max_gib",
        "MEMORY_LIMIT_FRACTION = 0.95",
        "RAY_CLUSTER_LOG_DIR",
        "GroupedGemmGluSm100",
        "MAX_FUNCTIONAL_EVIDENCE_MATCHES",
        '"Run three-update EP${EXPERT_MODEL_PARALLEL_SIZE} CuTeDSL ON functional arm"',
        'len(completed_ranks) != int(os.environ["TRAINING_GPU_COUNT"])',
    )
    for fragment in required:
        assert fragment in source, fragment
    assert (
        'if [[ "${FUNCTIONAL_GATE}" == "0" && "${PROFILE_ENABLED}" == "1" ]]; then'
        in source
    )


def test_timing_summarizer_accepts_bounded_live_workload_equivalence(
    tmp_path: Path,
) -> None:
    result = _run_timing_summarizer(tmp_path, on_token_scale=1.005)

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "results/timing_summary.json").read_text())
    equivalence = summary["workload_equivalence"]
    assert equivalence["observed"] is True
    assert equivalence["exact_observed_invariants"]["observed"] is True
    assert equivalence["prompt_sequence_identity_verified"] is False
    assert equivalence["limits"] == {
        "arm_total_relative_delta": 0.01,
        "paired_step_relative_delta": 0.02,
    }


def test_timing_summarizer_rejects_out_of_bounds_live_workload(
    tmp_path: Path,
) -> None:
    result = _run_timing_summarizer(tmp_path, on_token_scale=1.03)

    assert result.returncode != 0
    assert "measured workload equivalence failed" in result.stderr
    summary = json.loads((tmp_path / "results/timing_summary.json").read_text())
    assert summary["workload_equivalence"]["observed"] is False


def test_timing_summarizer_marks_single_arm_workload_equivalence_not_applicable(
    tmp_path: Path,
) -> None:
    result = _run_timing_summarizer(
        tmp_path,
        on_token_scale=1.0,
        timing_order=("on",),
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "results/timing_summary.json").read_text())
    equivalence = summary["workload_equivalence"]
    assert equivalence["required"] is False
    assert equivalence["observed"] is None
    assert equivalence["not_applicable_reason"] == "single timing arm"


def test_functional_summarizer_rejects_offload_sequence_prefix(
    tmp_path: Path,
) -> None:
    result = _run_functional_summarizer(tmp_path, offload_sequence=20)

    assert result.returncode != 0
    assert "functional offload telemetry requires" in result.stderr
    assert not (tmp_path / "results" / "functional_gate_summary.json").exists()


def test_functional_summarizer_rejects_initial_stale_generation_sequence(
    tmp_path: Path,
) -> None:
    result = _run_functional_summarizer(tmp_path, offload_sequence=2)

    assert result.returncode != 0
    assert "offload_sequence=3" in result.stderr
    assert not (tmp_path / "results" / "functional_gate_summary.json").exists()


def test_functional_summarizer_accepts_first_post_update_offload_sequence(
    tmp_path: Path,
) -> None:
    result = _run_functional_summarizer(tmp_path, offload_sequence=3)

    assert result.returncode == 0, result.stderr
    summary = json.loads(
        (tmp_path / "results" / "functional_gate_summary.json").read_text()
    )
    assert summary["offload_memory_evidence"]["completed_global_ranks"] == list(
        range(8)
    )
    assert summary["offload_memory_evidence"]["required_offload_sequence"] == 3


def test_functional_summarizer_accepts_finite_cgroup_fraction_below_limit(
    tmp_path: Path,
) -> None:
    result = _run_functional_summarizer(
        tmp_path,
        offload_sequence=3,
        cgroup_memory_peak_gib="94.999",
        cgroup_memory_max_gib="100.000",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(
        (tmp_path / "results" / "functional_gate_summary.json").read_text()
    )
    memory = summary["offload_memory_evidence"]["cgroup_memory"]
    assert memory["limit_fraction_exclusive"] == 0.95
    assert memory["finite_limit_global_ranks"] == list(range(8))
    assert memory["unavailable_limit_global_ranks"] == []


@pytest.mark.parametrize("peak", ["95.000", "95.001"])
def test_functional_summarizer_rejects_cgroup_fraction_at_or_above_limit(
    tmp_path: Path, peak: str
) -> None:
    result = _run_functional_summarizer(
        tmp_path,
        offload_sequence=3,
        cgroup_memory_peak_gib=peak,
        cgroup_memory_max_gib="100.000",
    )

    assert result.returncode != 0
    assert "cgroup memory peak/limit must be < 0.95" in result.stderr
    assert not (tmp_path / "results" / "functional_gate_summary.json").exists()


def test_functional_summarizer_classifies_unavailable_cgroup_limit(
    tmp_path: Path,
) -> None:
    result = _run_functional_summarizer(
        tmp_path,
        offload_sequence=3,
        cgroup_memory_peak_gib="unavailable",
        cgroup_memory_max_gib="unavailable",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(
        (tmp_path / "results" / "functional_gate_summary.json").read_text()
    )
    memory = summary["offload_memory_evidence"]["cgroup_memory"]
    assert memory["finite_limit_global_ranks"] == []
    assert memory["unavailable_limit_global_ranks"] == list(range(8))
    assert summary["post_job_slurm_accounting_required"] is True


def test_functional_summarizer_strips_ray_ansi_dedup_suffix(
    tmp_path: Path,
) -> None:
    result = _run_functional_summarizer(
        tmp_path,
        offload_sequence=3,
        cgroup_memory_peak_gib="94.999",
        cgroup_memory_max_gib="100.000",
        evidence_suffix="\x1b[32m [repeated 7x across cluster]\x1b[0m",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(
        (tmp_path / "results" / "functional_gate_summary.json").read_text()
    )
    matches = summary["offload_memory_evidence"]["matches"]
    assert matches
    assert all("\x1b" not in match["line"] for match in matches)


def test_functional_summarizer_ignores_ray_control_plane_log_fanout(
    tmp_path: Path,
) -> None:
    ray_logs = {
        f"session/logs/events/event_EXPORT_TASK_{index}.log": "control plane\n"
        for index in range(600)
    }
    ray_logs["session/logs/worker-acde-01000000-42.out"] = "policy worker\n"

    result = _run_functional_summarizer(
        tmp_path,
        offload_sequence=3,
        ray_logs=ray_logs,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(
        (tmp_path / "results" / "functional_gate_summary.json").read_text()
    )
    assert summary["evidence_scan"]["files_scanned"] == 2


def test_functional_summarizer_scales_worker_file_bound_for_sixteen_ranks(
    tmp_path: Path,
) -> None:
    ray_logs = {
        f"session/logs/worker-{index:08x}-01000000-{index}.out": "policy worker\n"
        for index in range(600)
    }

    result = _run_functional_summarizer(
        tmp_path,
        offload_sequence=3,
        ray_logs=ray_logs,
        training_gpu_count=16,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(
        (tmp_path / "results" / "functional_gate_summary.json").read_text()
    )
    assert summary["evidence_scan"]["files_scanned"] == 601
    assert summary["offload_memory_evidence"]["completed_global_ranks"] == list(
        range(16)
    )


@pytest.mark.parametrize(
    ("constant_overrides", "ray_logs", "evidence_suffix", "reason"),
    [
        (
            {"MAX_FUNCTIONAL_EVIDENCE_FILES": 1},
            {
                "worker-a-01000000-1.out": "first\n",
                "worker-b-01000000-2.err": "second\n",
            },
            "",
            "file_count_limit",
        ),
        (
            {"MAX_FUNCTIONAL_EVIDENCE_BYTES_PER_FILE": 64},
            {},
            " padding=" + "x" * 128,
            "per_file_tail_limit",
        ),
        (
            {"MAX_FUNCTIONAL_EVIDENCE_BYTES": 128},
            {},
            "",
            "total_byte_limit",
        ),
        (
            {"MAX_FUNCTIONAL_EVIDENCE_MATCHES": 1},
            {},
            "",
            "match_count_limit",
        ),
        (
            {"MAX_FUNCTIONAL_EVIDENCE_LINE_CHARS": 96},
            {},
            " padding=" + "x" * 128,
            "retained_line_limit",
        ),
    ],
)
def test_functional_summarizer_rejects_any_bounded_scan_truncation(
    tmp_path: Path,
    constant_overrides: dict[str, int],
    ray_logs: dict[str, str],
    evidence_suffix: str,
    reason: str,
) -> None:
    result = _run_functional_summarizer(
        tmp_path,
        offload_sequence=3,
        ray_logs=ray_logs,
        constant_overrides=constant_overrides,
        evidence_suffix=evidence_suffix,
    )

    assert result.returncode != 0
    assert "functional evidence scan was truncated" in result.stderr
    assert reason in result.stderr
    assert not (tmp_path / "results" / "functional_gate_summary.json").exists()


def test_functional_payload_does_not_emit_timing_or_profile_artifacts() -> None:
    source = MATRIX_PAYLOAD.read_text()
    required = (
        '[[ ! -e "${CONTAINER_RESULT_DIR}/timing_summary.json" ]]',
        '[[ ! -e "${CONTAINER_RESULT_DIR}/profiles" ]]',
        '[[ ! -e "${CONTAINER_RESULT_DIR}/kernel_attribution.json" ]]',
        '[[ ! -e "${CONTAINER_RESULT_DIR}/feature_attribution.json" ]]',
    )
    for fragment in required:
        assert fragment in source, fragment


def test_matrix_payload_fails_closed_on_missing_feature_implementations() -> None:
    source = MATRIX_PAYLOAD.read_text()
    required = (
        'NEMO2606_FULL_CG_ENABLED="${NEMO2606_FULL_CG_ENABLED:-0}"',
        'NEMO2606_A2A_ENABLED="${NEMO2606_A2A_ENABLED:-0}"',
        "nemo_rl/models/megatron/full_cuda_graph.py",
        "build_full_cuda_graph_schedule",
        "return_schedule_plan",
        "overlap_moe_expert_parallel_comm",
        "policy.megatron_cfg.cuda_graph_impl=full_iteration",
        "policy.megatron_cfg.overlap_moe_expert_parallel_comm=true",
        "policy.megatron_cfg.high_priority_a2a_comm_stream=true",
        "policy.megatron_cfg.delay_wgrad_compute=true",
        '"feature_context"',
        '"full_cg_enabled"',
        '"a2a_enabled"',
        '"context_single_arm" if expected_full_cg else "context_local_cutedsl_pair"',
        '"cross_context_factorial_aggregate_available": False',
        '"kernel_attribution"',
    )
    for fragment in required:
        assert fragment in source, fragment
    assert '-newer "${profile_marker}"' in source
    assert "CUDA_GRAPH_WARMUP_STEPS=3" in source
    assert "profile_max_steps=$((CUDA_GRAPH_WARMUP_STEPS + 2))" in source
    assert (
        'profile_step_range="$((CUDA_GRAPH_WARMUP_STEPS + 1)):$((CUDA_GRAPH_WARMUP_STEPS + 2))"'
        in source
    )
    assert 'export NRL_NSYS_PROFILE_STEP_RANGE="${profile_step_range}"' in source


def test_shell_entrypoints_are_parseable() -> None:
    for path in (
        SUBMITTER,
        OFFICIAL_SUBMITTER,
        DIRECT_AB_SUBMITTER,
        MATRIX_PAYLOAD,
    ):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"
