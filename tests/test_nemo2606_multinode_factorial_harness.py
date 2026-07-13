import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
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


def _run_functional_summarizer(
    tmp_path: Path,
    offload_sequence: int,
    *,
    cgroup_memory_peak_gib: str = "unavailable",
    cgroup_memory_max_gib: str = "unavailable",
    ray_logs: dict[str, str] | None = None,
    constant_overrides: dict[str, int] | None = None,
    evidence_suffix: str = "",
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
        for rank in range(8)
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
            "TRAINING_GPU_COUNT": "8",
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


def test_functional_submitter_exports_one_fail_closed_job(tmp_path: Path) -> None:
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
        ["bash", str(SUBMITTER), "--test-only"],
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
    assert call["nodes"] == "2"
    assert call["segment_size"] == "2"
    assert call["gpus_per_node"] == "4"
    assert call["full_cg"] == "0"
    assert call["a2a"] == "0"
    assert "--nodes=2" in call["argv"]
    assert "--segment=2" in call["argv"]
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


def test_functional_payload_uses_effective_segment_and_one_arm_manifest() -> None:
    source = MATRIX_PAYLOAD.read_text()
    required = (
        'BENCHMARK_SEGMENT_SIZE="${CUTEDSL_BENCHMARK_SEGMENT_SIZE:-${CUTEDSL_SEGMENT:-1}}"',
        '"cluster.segment_size=${BENCHMARK_SEGMENT_SIZE}"',
        '"functional_gate": os.environ["FUNCTIONAL_GATE"] == "1"',
        '"performance_eligible": os.environ["FUNCTIONAL_GATE"] != "1"',
        '"segment_size": cluster_config["segment_size"]',
        '"segment": int(os.environ["BENCHMARK_SEGMENT_SIZE"])',
        'if os.environ["FUNCTIONAL_GATE"] != "1":',
        'fixed_config_evidence = {"on": fixed_config_by_arm["on"]}',
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
