import copy
import json
import os
import re
import subprocess
import sys
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
    assert f"--nodes={expected_nodes}" in call["argv"]
    assert f"--segment={expected_segment_size}" in call["argv"]
    assert "--test-only" in call["argv"]


def test_matrix_payload_reuses_collectors_in_existing_ray_mode() -> None:
    source = MATRIX_PAYLOAD.read_text()
    assert 'EXISTING_RAY="${CUTEDSL_BENCHMARK_EXISTING_RAY:-0}"' in source
    assert 'if [[ "${EXISTING_RAY}" == "1" ]]; then' in source
    assert "SRUN=()" in source
    assert (
        'NODE_LOCAL_WORKER_VENV_ROOT="/tmp/${USER}/nemo2606-factorial/${RUN_ID}/worker_venvs"'
        in source
    )
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
        "fixed_config_evidence = {arm: fixed_config_by_arm[arm] for arm in expected_arms}",
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
    for path in (SUBMITTER, OFFICIAL_SUBMITTER, MATRIX_PAYLOAD):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"
