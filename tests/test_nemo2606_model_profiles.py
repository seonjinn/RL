import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
PROFILE_MODULE = EXPERIMENT_DIR / "lib/model_profile.py"
PROFILE_DIR = EXPERIMENT_DIR / "model_profiles"
SUBMITTER = EXPERIMENT_DIR / "submit_nemo2606_2n4g_factorial.sh"
MATRIX_PAYLOAD = EXPERIMENT_DIR / "run_cutedsl_matrix.sbatch"


def _load_profile_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("model_profile", PROFILE_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    (
        "profile_name",
        "expected_profile_id",
        "expected_model",
        "expected_topology",
        "expected_workload",
        "expected_rollout",
    ),
    (
        (
            "qwen3_30ba3b_2n4g.json",
            "qwen3_30ba3b_2n4g",
            "Qwen/Qwen3-30B-A3B",
            {
                "num_nodes": 2,
                "gpus_per_node": 4,
                "segment_size": None,
                "tp": 1,
                "pp": 1,
                "vpp": None,
                "num_layers_in_first_pipeline_stage": None,
                "num_layers_in_last_pipeline_stage": None,
                "cp": 1,
                "ep": 4,
                "etp": 1,
            },
            {
                "train_global_batch_size": 8,
                "train_micro_batch_size": 1,
                "logprob_batch_size": 1,
                "max_total_sequence_length": 1024,
                "sequence_packing_enabled": False,
                "num_prompts_per_step": 4,
                "num_generations_per_prompt": 2,
            },
            {"precision": "fp8", "tensor_parallel_size": 1},
        ),
        (
            "qwen3_30ba3b_4n4g.json",
            "qwen3_30ba3b_4n4g",
            "Qwen/Qwen3-30B-A3B",
            {
                "num_nodes": 4,
                "gpus_per_node": 4,
                "segment_size": 4,
                "tp": 1,
                "pp": 1,
                "vpp": None,
                "num_layers_in_first_pipeline_stage": None,
                "num_layers_in_last_pipeline_stage": None,
                "cp": 1,
                "ep": 16,
                "etp": 1,
            },
            {
                "train_global_batch_size": 2048,
                "train_micro_batch_size": 1,
                "logprob_batch_size": 2,
                "max_total_sequence_length": 4096,
                "sequence_packing_enabled": True,
                "num_prompts_per_step": 64,
                "num_generations_per_prompt": 32,
            },
            {"precision": "bfloat16", "tensor_parallel_size": 1},
        ),
        (
            "qwen3_235b_16n4g.json",
            "qwen3_235b_16n4g",
            "Qwen/Qwen3-235B-A22B",
            {
                "num_nodes": 16,
                "gpus_per_node": 4,
                "segment_size": 16,
                "tp": 2,
                "pp": 4,
                "vpp": None,
                "num_layers_in_first_pipeline_stage": 23,
                "num_layers_in_last_pipeline_stage": 23,
                "cp": 2,
                "ep": 16,
                "etp": 1,
            },
            {
                "train_global_batch_size": 512,
                "train_micro_batch_size": 1,
                "logprob_batch_size": 1,
                "max_total_sequence_length": 8192,
                "sequence_packing_enabled": True,
                "num_prompts_per_step": 16,
                "num_generations_per_prompt": 32,
            },
            {"precision": "bfloat16", "tensor_parallel_size": 8},
        ),
        (
            "qwen3_235b_16n4g_a2a_vpp2.json",
            "qwen3_235b_16n4g_a2a_vpp2",
            "Qwen/Qwen3-235B-A22B",
            {
                "num_nodes": 16,
                "gpus_per_node": 4,
                "segment_size": 16,
                "tp": 2,
                "pp": 4,
                "vpp": 2,
                "num_layers_in_first_pipeline_stage": 22,
                "num_layers_in_last_pipeline_stage": 24,
                "cp": 2,
                "ep": 16,
                "etp": 1,
            },
            {
                "train_global_batch_size": 512,
                "train_micro_batch_size": 1,
                "logprob_batch_size": 1,
                "max_total_sequence_length": 8192,
                "sequence_packing_enabled": True,
                "num_prompts_per_step": 16,
                "num_generations_per_prompt": 32,
            },
            {"precision": "bfloat16", "tensor_parallel_size": 8},
        ),
    ),
)
def test_model_profiles_are_exact_and_validate_resolved_recipes(
    profile_name: str,
    expected_profile_id: str,
    expected_model: str,
    expected_topology: dict[str, int | None],
    expected_workload: dict[str, int | bool],
    expected_rollout: dict[str, str | int],
) -> None:
    module = _load_profile_module()
    profile = module.load_model_profile(PROFILE_DIR / profile_name)

    assert profile.profile_id == expected_profile_id
    assert profile.artifacts.model_repo_id == expected_model
    assert profile.artifacts.dataset_repo_id == "nvidia/OpenMathInstruct-2"
    assert profile.artifacts.dataset_split == "train_1M"
    assert profile.artifacts.dataset_num_rows == 1_000_000
    assert profile.topology.model_dump() == expected_topology
    assert profile.workload.model_dump() == expected_workload
    assert {
        "precision": profile.runtime.rollout_precision,
        "tensor_parallel_size": profile.runtime.generation_tensor_parallel_size,
    } == expected_rollout
    assert profile.runtime.policy_precision == "bfloat16"
    expected_policy_gpus = {
        "qwen3_30ba3b_2n4g.json": 4,
        "qwen3_30ba3b_4n4g.json": 16,
    }.get(profile_name, 64)
    assert profile.runtime.policy_training_gpu_count == expected_policy_gpus
    assert profile.runtime.generation_colocated is (
        profile_name != "qwen3_30ba3b_2n4g.json"
    )
    if profile_name == "qwen3_235b_16n4g_a2a_vpp2.json":
        assert profile.default_contexts == "g0a0,g0a1"
        assert profile.runtime.allow_full_cg is False
        assert profile.runtime.allow_a2a is True
        assert profile.runtime.activation_checkpointing is True
        assert profile.runtime.recompute_granularity == "selective"
        assert profile.runtime.recompute_method is None
        assert profile.runtime.recompute_modules is None
    assert profile.provenance.triton_cache_scope == "job_node_local"
    assert profile.provenance.megatron_checkpoint_scope == "job_shared"
    assert profile.provenance.megatron_checkpoint_marker == (
        "iter_0000000/run_config.yaml"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", module.profile_sha256(profile))

    resolved = module.validate_resolved_recipe(profile, PROJECT_ROOT)
    assert resolved["model_name"] == expected_model
    assert resolved["topology"] == expected_topology
    assert resolved["workload"] == expected_workload


def test_profile_loader_and_resolver_fail_closed(tmp_path: Path) -> None:
    module = _load_profile_module()
    original = json.loads((PROFILE_DIR / "qwen3_235b_16n4g.json").read_text())

    original["unknown"] = True
    invalid = tmp_path / "unknown.json"
    invalid.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        module.load_model_profile(invalid)

    original.pop("unknown")
    original["topology"]["tp"] = 1
    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(json.dumps(original))
    profile = module.load_model_profile(mismatched)
    with pytest.raises(ValueError, match="resolved topology does not match profile"):
        module.validate_resolved_recipe(profile, PROJECT_ROOT)


def test_profile_shell_exports_include_immutable_identity() -> None:
    module = _load_profile_module()
    profile_path = PROFILE_DIR / "qwen3_235b_16n4g.json"
    profile = module.load_model_profile(profile_path)

    exports = module.shell_exports(profile, profile_path)

    assert exports["CUTEDSL_MODEL_PROFILE_ID"] == "qwen3_235b_16n4g"
    assert exports["CUTEDSL_MODEL_PROFILE_SHA256"] == module.profile_sha256(profile)
    assert exports["CUTEDSL_MODEL_REPO_ID"] == "Qwen/Qwen3-235B-A22B"
    assert exports["CUTEDSL_DATASET_REPO_ID"] == "nvidia/OpenMathInstruct-2"
    assert exports["CUTEDSL_MEGATRON_CHECKPOINT_SCOPE"] == "job_shared"
    assert exports["CUTEDSL_MEGATRON_CHECKPOINT_MARKER"] == (
        "iter_0000000/run_config.yaml"
    )
    assert exports["CUTEDSL_PROFILE_RECIPE"].endswith(
        "grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml"
    )
    assert exports["CUTEDSL_PROFILE_NUM_NODES"] == "16"
    assert exports["CUTEDSL_PROFILE_TP"] == "2"
    assert exports["CUTEDSL_PROFILE_PP"] == "4"
    assert exports["CUTEDSL_PROFILE_CP"] == "2"
    assert exports["CUTEDSL_PROFILE_EP"] == "16"
    assert exports["CUTEDSL_PROFILE_TRAIN_GLOBAL_BATCH_SIZE"] == "512"
    assert exports["CUTEDSL_PROFILE_MAX_TOTAL_SEQUENCE_LENGTH"] == "8192"


def _write_mock_sbatch(mock_bin: Path, calls_path: Path) -> None:
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
    "argv": [arg for arg in sys.argv[1:] if not arg.startswith("--export-file=")],
    "profile_id": payload["CUTEDSL_MODEL_PROFILE_ID"],
    "profile_sha256": payload["CUTEDSL_MODEL_PROFILE_SHA256"],
    "profile_path": payload["CUTEDSL_MODEL_PROFILE_PATH"],
    "recipe": payload["CUTEDSL_BENCHMARK_RECIPE"],
    "nodes": payload["CUTEDSL_BENCHMARK_NUM_NODES"],
    "gpus_per_node": payload["CUTEDSL_BENCHMARK_GPUS_PER_NODE"],
    "segment_size": payload["CUTEDSL_BENCHMARK_SEGMENT_SIZE"],
    "config_segment_size": payload["CUTEDSL_BENCHMARK_CONFIG_SEGMENT_SIZE"],
    "train_global_batch_size": payload["CUTEDSL_BENCHMARK_TRAIN_GLOBAL_BATCH_SIZE"],
    "expert_model_parallel_size": payload["CUTEDSL_BENCHMARK_EXPERT_MODEL_PARALLEL_SIZE"],
    "context": payload["NEMO2606_FACTORIAL_CONTEXT"],
    "a2a_enabled": payload["NEMO2606_A2A_ENABLED"],
    "checkpoint_scope": payload["CUTEDSL_MEGATRON_CHECKPOINT_SCOPE"],
    "checkpoint_marker": payload["CUTEDSL_MEGATRON_CHECKPOINT_MARKER"],
}
with Path(os.environ["MOCK_SBATCH_CALLS"]).open("a") as output:
    output.write(json.dumps(record) + "\\n")
print("mock-job")
"""
    )
    mock_sbatch.chmod(0o755)


def test_qwen30_null_config_segment_uses_positive_scheduler_segment(
    tmp_path: Path,
) -> None:
    module = _load_profile_module()
    profile_path = PROFILE_DIR / "qwen3_30ba3b_2n4g.json"
    profile = module.load_model_profile(profile_path)
    exports = module.shell_exports(profile, profile_path)

    assert exports["CUTEDSL_PROFILE_SEGMENT_SIZE"] == "2"
    assert exports["CUTEDSL_PROFILE_CONFIG_SEGMENT_SIZE"] == "null"

    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    _write_mock_sbatch(mock_bin, calls_path)
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
        ["bash", str(SUBMITTER), "--model-profile", str(profile_path), "--test-only"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["segment_size"] == "2"
    assert calls[0]["config_segment_size"] == "null"
    assert "--segment=2" in calls[0]["argv"]


def test_qwen235_test_only_validates_and_exports_resolved_profile(
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    _write_mock_sbatch(mock_bin, calls_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--model-profile",
            str(PROFILE_DIR / "qwen3_235b_16n4g.json"),
            "--test-only",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 3
    for call in calls:
        assert call["profile_id"] == "qwen3_235b_16n4g"
        assert re.fullmatch(r"[0-9a-f]{64}", call["profile_sha256"])
        assert call["profile_path"] == str(
            (PROFILE_DIR / "qwen3_235b_16n4g.json").resolve()
        )
        assert call["recipe"].endswith(
            "grpo-qwen3-235b-16n4g-megatron-mxfp8-cutedsl.yaml"
        )
        assert call["nodes"] == "16"
        assert call["gpus_per_node"] == "4"
        assert call["segment_size"] == "16"
        assert call["train_global_batch_size"] == "512"
        assert call["expert_model_parallel_size"] == "16"
        assert call["checkpoint_scope"] == "job_shared"
        assert call["checkpoint_marker"] == "iter_0000000/run_config.yaml"
        assert "--nodes=16" in call["argv"]
        assert "--segment=16" in call["argv"]
        assert "--test-only" in call["argv"]


def test_test_only_rejects_profile_recipe_mismatch_before_sbatch(
    tmp_path: Path,
) -> None:
    payload = json.loads((PROFILE_DIR / "qwen3_235b_16n4g.json").read_text())
    payload["topology"]["tp"] = 1
    mismatched_profile = tmp_path / "mismatched.json"
    mismatched_profile.write_text(json.dumps(payload))
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    _write_mock_sbatch(mock_bin, calls_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--model-profile",
            str(mismatched_profile),
            "--test-only",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "resolved topology does not match profile" in result.stderr
    assert not calls_path.exists()


def test_qwen235_rejects_unsupported_feature_context_before_sbatch(
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    _write_mock_sbatch(mock_bin, calls_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
            "NEMO2606_FACTORIAL_CONTEXTS": "g0a1",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--model-profile",
            str(PROFILE_DIR / "qwen3_235b_16n4g.json"),
            "--test-only",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "selected profile does not allow A2A" in result.stderr
    assert not calls_path.exists()


def test_qwen235_a2a_vpp_profile_exports_only_eager_a2a_contexts(
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    _write_mock_sbatch(mock_bin, calls_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--model-profile",
            str(PROFILE_DIR / "qwen3_235b_16n4g_a2a_vpp2.json"),
            "--test-only",
        ],
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
    assert {call["a2a_enabled"] for call in calls if call["context"] == "g0a0"} == {"0"}
    assert {call["a2a_enabled"] for call in calls if call["context"] == "g0a1"} == {"1"}
    assert {call["profile_id"] for call in calls} == {"qwen3_235b_16n4g_a2a_vpp2"}


def test_qwen235_a2a_vpp_effective_config_keeps_selective_recompute() -> None:
    from nemo_rl.utils.config import load_config, parse_hydra_overrides
    from omegaconf import OmegaConf

    module = _load_profile_module()
    profile_path = PROFILE_DIR / "qwen3_235b_16n4g_a2a_vpp2.json"
    profile = module.load_model_profile(profile_path)
    exports = module.shell_exports(profile, profile_path)

    assert exports["CUTEDSL_PROFILE_VPP"] == "2"
    assert exports["CUTEDSL_PROFILE_FIRST_STAGE_LAYERS"] == "22"
    assert exports["CUTEDSL_PROFILE_LAST_STAGE_LAYERS"] == "24"

    config = parse_hydra_overrides(
        load_config(PROJECT_ROOT / profile.recipe),
        [
            "policy.megatron_cfg.overlap_moe_expert_parallel_comm=true",
            "policy.megatron_cfg.high_priority_a2a_comm_stream=true",
            "policy.megatron_cfg.delay_wgrad_compute=true",
        ],
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    policy = resolved["policy"]
    megatron = policy["megatron_cfg"]
    assert megatron["pipeline_model_parallel_size"] == 4
    assert megatron["virtual_pipeline_model_parallel_size"] == 2
    assert megatron["num_layers_in_first_pipeline_stage"] == 22
    assert megatron["num_layers_in_last_pipeline_stage"] == 24
    assert megatron["activation_checkpointing"] is True
    assert megatron["recompute_granularity"] == "selective"
    assert megatron["recompute_method"] is None
    assert megatron["recompute_modules"] is None
    assert megatron["overlap_moe_expert_parallel_comm"] is True


def test_qwen235_a2a_vpp_profile_rejects_full_cg_before_sbatch(
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    _write_mock_sbatch(mock_bin, calls_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
            "NEMO2606_FACTORIAL_CONTEXTS": "g1a0",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--model-profile",
            str(PROFILE_DIR / "qwen3_235b_16n4g_a2a_vpp2.json"),
            "--test-only",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "selected profile does not allow full-iteration CUDA Graph" in result.stderr
    assert not calls_path.exists()


def test_matrix_payload_binds_cache_config_and_checkpoint_to_profile() -> None:
    source = MATRIX_PAYLOAD.read_text()
    submitter_source = SUBMITTER.read_text()

    assert "local status=$?" in submitter_source
    assert 'exit "${status}"' in submitter_source
    assert 'MODEL_PROFILE_PATH="${CUTEDSL_MODEL_PROFILE_PATH:?' in source
    assert 'MODEL_PROFILE_SHA256="${CUTEDSL_MODEL_PROFILE_SHA256:?' in source
    assert '--model-profile "${MODEL_PROFILE_PATH}"' in source
    assert (
        'profile = json.loads(Path(os.environ["MODEL_PROFILE_PATH"]).read_text())'
        in source
    )
    assert 'assert profile_sha256 == os.environ["MODEL_PROFILE_SHA256"]' in source
    assert 'assert profile_topology == profile["topology"]' in source
    assert 'assert profile_workload == profile["workload"]' in source
    assert '"model_profile": profile' in source
    assert '"model_profile_sha256": profile_sha256' in source
    assert '"megatron_checkpoint_scope": "job_shared"' in source
    assert '"megatron_checkpoint_marker": profile["provenance"][' in source
    assert '"model": ("Qwen/Qwen3-30B-A3B", None)' not in source
