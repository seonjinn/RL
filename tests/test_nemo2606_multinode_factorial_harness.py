import json
import os
import re
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
SUBMITTER = EXPERIMENT_DIR / "submit_nemo2606_2n4g_factorial.sh"
MATRIX_PAYLOAD = EXPERIMENT_DIR / "run_cutedsl_matrix.sbatch"
PROFILE_LOADER = EXPERIMENT_DIR / "lib/cluster_profile.sh"
RAY_SUB = PROJECT_ROOT / "ray.sub"
RECIPE = (
    PROJECT_ROOT
    / "examples/configs/recipes/llm/performance"
    / "grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-factorial.yaml"
)

register_omegaconf_resolvers()


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


def test_submitter_launches_real_two_node_ray_sub_cluster() -> None:
    source = SUBMITTER.read_text()
    required = (
        'readonly RAY_SUB="${REPO_ROOT}/ray.sub"',
        '"--nodes=2"',
        '"CONTAINER=${CUTEDSL_IMAGE}"',
        '"GPUS_PER_NODE=4"',
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


def test_submitter_covers_four_contexts_with_alternating_replicas() -> None:
    source = SUBMITTER.read_text()
    assert 'CONTEXTS="${NEMO2606_FACTORIAL_CONTEXTS:-g0a0,g1a0,g0a1,g1a1}"' in source
    assert 'REPLICATES="${NEMO2606_FACTORIAL_REPLICATES:-3}"' in source
    assert 'WARMUP_UPDATES="${NEMO2606_FACTORIAL_WARMUP_UPDATES:-5}"' in source
    assert 'MEASURED_UPDATES="${NEMO2606_FACTORIAL_MEASURED_UPDATES:-20}"' in source
    assert "((REPLICATES < 3))" in source
    assert 'timing_order="on,off"' in source
    assert 'timing_order="off,on"' in source
    assert '"NEMO2606_FULL_CG_ENABLED=${full_cg_enabled}"' in source
    assert '"NEMO2606_A2A_ENABLED=${a2a_enabled}"' in source
    assert '"CUTEDSL_BENCHMARK_ORDER=${timing_order}"' in source
    assert 'if [[ "${TEST_ONLY}" == "0" && "${needs_a2a}" == "1" ]]' in source
    assert 'if [[ "${TEST_ONLY}" == "0" && "${needs_full_cg}" == "1" ]]' in source
    assert source.index('needs_full_cg="0"') < source.index("job_id=$(sbatch")


def test_submitter_test_only_exports_twelve_ray_jobs(tmp_path: Path) -> None:
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
    "existing_ray": payload["CUTEDSL_BENCHMARK_EXISTING_RAY"],
    "nodes": payload["CUTEDSL_BENCHMARK_NUM_NODES"],
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
    assert len(calls) == 12
    assert [call["context"] for call in calls] == [
        context for context in ("g0a0", "g1a0", "g0a1", "g1a1") for _ in range(3)
    ]
    for context in ("g0a0", "g1a0", "g0a1", "g1a1"):
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
        "policy.megatron_cfg.expert_model_parallel_size=${TRAINING_GPU_COUNT}" in source
    )
    assert "policy.train_global_batch_size=$((TRAINING_GPU_COUNT * 2))" in source
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
        '"aggregation_scope": "context_local_cutedsl_pair"',
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
    for path in (SUBMITTER, MATRIX_PAYLOAD):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"
