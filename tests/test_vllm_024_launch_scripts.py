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

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DYNAMICSD_LAUNCHER = (
    REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_eagle3_dynamicsd_step20.sh"
)
CG_TOP_P_PROFILE_LAUNCHER = (
    REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_cg_top_p_refit_profile.sh"
)
LONG_OUTPUT_LAUNCHER = (
    REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_long_output_matrix.sh"
)
HF_PREWARM_LAUNCHER = (
    REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_hf_snapshot_prewarm.sh"
)
SUBMISSIONS_HEADER = (
    "timestamp\tmodel\tvariant\tjob_id\tnodes\tsegment\tcommit\t"
    "wandb_run_id\twandb_url\trecipe\tdraft_model\tcontainer\t"
    "container_sha256\tmax_steps\tstatic_k\tdynamic_schedule\t"
    "rejection_sample_method\tdraft_sample_method\tmax_num_batched_tokens\t"
    "max_num_seqs\toutput_max_model_len\tspecdec_context_headroom_tokens\t"
    "max_cudagraph_capture_size\tcudagraph_capture_sizes\t"
    "num_prompts_per_step\t"
    "num_generations_per_prompt\ttrain_global_batch_size\t"
    "max_total_sequence_length\tmax_new_tokens\tcommand"
)
PARITY_LAUNCHER = (
    REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_generation_parity.sh"
)


def _run_script(path: Path, *args: str, **environment: str) -> str:
    result = subprocess.run(
        ["bash", str(path), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **environment},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _run_script_unchecked(
    path: Path, *args: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(path), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )


def _prepare_submit_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    tracked_files = (
        checkout
        / "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        checkout / "experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh",
        checkout / "ray.sub",
    )
    for tracked_file in tracked_files:
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.touch()

    git_environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "contract-test",
        "GIT_AUTHOR_EMAIL": "contract-test@example.invalid",
        "GIT_COMMITTER_NAME": "contract-test",
        "GIT_COMMITTER_EMAIL": "contract-test@example.invalid",
    }
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "contract-test"],
        ["git", "config", "user.email", "contract-test@example.invalid"],
        ["git", "add", "."],
        ["git", "commit", "-q", "-m", "contract-test"],
    ):
        subprocess.run(
            command,
            cwd=checkout,
            env=git_environment,
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        ["git", "branch", "-M", "main"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    (stub_bin / "sbatch").write_text(
        "#!/usr/bin/env bash\nprintf '12345\\n'\n", encoding="utf-8"
    )
    (stub_bin / "readlink").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$2\"\n", encoding="utf-8"
    )
    (stub_bin / "date").write_text(
        "#!/usr/bin/env bash\nprintf '2026-07-09T00:00:00+00:00\\n'\n",
        encoding="utf-8",
    )
    for stub in stub_bin.iterdir():
        stub.chmod(0o755)

    container = tmp_path / "container.sqsh"
    container.touch()
    draft_model = tmp_path / "draft-model"
    draft_model.mkdir()
    experiment_root = tmp_path / "runs"
    environment = {
        "REPO_DIR": str(checkout),
        "CONTAINER": str(container),
        "QWEN30_DRAFT_MODEL": str(draft_model),
        "EXPERIMENT_ROOT": str(experiment_root),
        "RUN_TAG": "submit-contract",
        "ATTEMPT_ID": "attempt-1",
        "WANDB_API_KEY": "contract-test-key",
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
    }
    return environment, experiment_root / "submissions.tsv"


def _dry_run_dynamicsd(model: str, variant: str) -> str:
    return _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        model,
        variant,
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )


def _dry_run_parity(variant: str, mode: str) -> str:
    return _run_script(
        PARITY_LAUNCHER,
        "dry-run",
        variant,
        mode,
        REPO_DIR="/lustre/users/sna/RL",
        LYRIS_ROOT="/lustre/users/sna",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="parity-contract-test",
        TARGET_MODEL="/lustre/users/sna/qwen32",
        DRAFT_MODEL="/lustre/users/sna/qwen32-eagle3",
    )


def test_compat_smoke_uses_short_compute_node_tmpdir() -> None:
    output = _run_script(
        REPO_ROOT / "scripts" / "submit_vllm_024_compat_smoke.sh",
        REPO_DIR=str(REPO_ROOT),
        CONTAINER="/unused/nemo-rl.sqsh",
        DRY_RUN="true",
    )

    assert "TMPDIR=/tmp" in output


def test_performance_launcher_uses_short_compute_node_tmpdir() -> None:
    output = _run_script(
        REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
    )

    assert "TMPDIR=/tmp" in output


def test_hf_snapshot_prewarm_matches_lyris_topology_and_dflash_checkpoint() -> None:
    output = _run_script(
        HF_PREWARM_LAUNCHER,
        "dry-run",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        EXPERIMENT_ROOT="/lustre/users/sna/experiments/dflash-prewarm",
    )

    assert "inference-optimization/Qwen3-30B-A3B-speculator.dflash" in output
    assert "cache_dir=/lustre/users/sna/hf_home/hub" in output
    assert "--nodes=1" in output
    assert "--segment=1" in output
    assert "--gres" not in output


def test_performance_launcher_preserves_compute_visible_workdir() -> None:
    output = _run_script(
        REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        REPO_DIR="/lustre/users/sna/RL",
    )

    assert "CONTAINER_WORKDIR=/lustre/users/sna/RL" in output


def test_performance_launcher_imports_nemo_rl_from_the_checkout() -> None:
    output = _run_script(
        REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        REPO_DIR="/lustre/users/sna/RL",
    )

    assert "PYTHONPATH=/lustre/users/sna/RL" in output


def test_performance_launcher_uses_node_local_compiler_caches() -> None:
    output = _run_script(
        REPO_ROOT / "experiments" / "vllm_024_upgrade" / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        RUN_TAG="cache-test",
    )

    assert "TRITON_CACHE_DIR=/tmp/nemorl-vllm024-triton-cache-test-qwen32b" in output
    assert (
        "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-vllm024-inductor-cache-test-qwen32b"
        in output
    )


def test_ray_launcher_accepts_an_explicit_container_workdir() -> None:
    source = (REPO_ROOT / "ray.sub").read_text(encoding="utf-8")

    assert "CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$SLURM_SUBMIT_DIR}" in source
    assert 'COMMON_SRUN_ARGS+=" --container-workdir=$CONTAINER_WORKDIR"' in source


def test_ray_launcher_completes_final_distributed_log_sync_before_driver_exit() -> None:
    source = (REPO_ROOT / "ray.sub").read_text(encoding="utf-8")

    driver_exit = source.index("driver_exit_code=\\$?")
    request = source.index(".ray_logs_final_sync_requested", driver_exit)
    completion = source.index(".ray_logs_final_sync_complete", request)
    final_status = source.index("final_sync_exit_status", completion)
    assert driver_exit < request < completion < final_status
    assert ".ray_logs_final_sync_ack.head" in source
    assert ".ray_logs_final_sync_ack.worker-\\$SLURM_PROCID" in source
    assert "expected_sync_nodes=(head)" in source
    assert 'expected_sync_acks="\\${#expected_sync_nodes[@]}"' in source


@pytest.mark.parametrize("assignment", ["head_cmd", "worker_cmd"])
def test_ray_launcher_generated_scripts_remain_valid_bash(assignment: str) -> None:
    source = (REPO_ROOT / "ray.sub").read_text(encoding="utf-8")
    start_marker = f"{assignment}=$(cat <<EOF\n"
    start = source.index(start_marker) + len(start_marker)
    generated_script = source[start : source.index("\nEOF\n)", start)].replace(
        "\\$", "$"
    )

    subprocess.run(
        ["bash", "-n"],
        input=generated_script,
        text=True,
        check=True,
        capture_output=True,
    )


def test_dynamicsd_launcher_preserves_matched_runtime_contract() -> None:
    output = _dry_run_dynamicsd("qwen32b", "dynamic")

    assert "grpo.max_num_steps=20" in output
    assert "checkpointing.enabled=false" in output
    assert "policy.generation.vllm_cfg.enforce_eager=false" in output
    assert "policy.generation.temperature=1.0" in output
    assert "policy.generation.top_p=1.0" in output
    assert "compilation_config.cudagraph_mode=FULL_AND_PIECEWISE" in output
    assert "NRL_VLLM_ENABLE_V2_DRAFT_DECODE_CAPTURE_PROFILE=true" in output
    assert "cluster.segment_size=4" in output
    assert "--nodes=4" in output
    assert "--segment=4" in output
    assert "--dependency=" in output
    assert "--gres=gpu:4" in output
    assert "logger.wandb.entity=nvidia" in output
    assert "logger.tensorboard_enabled=false" in output
    assert "WANDB_RESUME=never" in output


def test_dynamicsd_launcher_honors_sampling_overrides() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen32b",
        "eagle3_k1",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="top-p-contract-test",
        ATTEMPT_ID="attempt-1",
        TEMPERATURE="1.0",
        TOP_P="0.7",
    )

    assert "policy.generation.temperature=1.0" in output
    assert "policy.generation.top_p=0.7" in output
    assert "policy.generation.top_p=1.0" not in output


def test_dynamicsd_launcher_keeps_target_model_and_tokenizer_matched() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen235b",
        "eagle3_k1",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="target-model-contract-test",
        ATTEMPT_ID="attempt-1",
        POLICY_MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )

    assert "policy.model_name=Qwen/Qwen3-235B-A22B-Instruct-2507" in output
    assert "policy.tokenizer.name=Qwen/Qwen3-235B-A22B-Instruct-2507" in output


def test_dynamicsd_launcher_propagates_refit_diagnostics_to_vllm_workers() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen32b",
        "eagle3_k1",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="refit-diagnostics-contract-test",
        ATTEMPT_ID="attempt-1",
        REFIT_DIAGNOSTICS="true",
    )

    assert "NRL_VLLM_REFIT_DIAGNOSTICS=true" in output
    assert "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=NRL_VLLM_REFIT_DIAGNOSTICS" in output


@pytest.mark.parametrize("top_p", ["0", "1.1", "not-a-number"])
def test_dynamicsd_launcher_rejects_invalid_top_p(top_p: str) -> None:
    result = _run_script_unchecked(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen32b",
        "baseline",
        TOP_P=top_p,
    )

    assert result.returncode == 2
    assert "TOP_P must be a number in (0, 1]" in result.stderr


def test_cg_top_p_profile_wrapper_renders_complete_matched_matrix() -> None:
    output = _run_script(
        CG_TOP_P_PROFILE_LAUNCHER,
        "dry-run",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="cg-top-p-contract-test",
        ATTEMPT_ID="attempt-1",
    )

    assert output.count("[PROFILE-MATRIX]") == 12
    for model in ("qwen30ba3b", "qwen32b", "qwen235b"):
        for top_p_label in ("top_p10", "top_p07"):
            for variant in ("baseline", "eagle3_k1"):
                assert f"model={model} top_p={top_p_label} variant={variant}" in output
    assert "max_cudagraph_capture_size=128" in output
    assert "max_cudagraph_capture_size=256" in output
    assert "max_cudagraph_capture_size=512" in output


def test_dynamicsd_launcher_rebuilds_the_vllm_024_worker_runtime() -> None:
    output = _dry_run_dynamicsd("qwen235b", "eagle3_k5")

    assert "/opt/nemo_rl_venv/bin/python" in output
    assert (
        "NEMO_RL_VENV_DIR=/tmp/nemorl-vllm024-venvs-contract-test-attempt-1-"
        "qwen235b-eagle3_k5"
    ) in output
    assert "NEMO_RL_VENV_DIR=/lustre" not in output
    assert "NRL_FORCE_REBUILD_VENVS=true" in output
    assert "uv run" not in output


def test_dynamicsd_launcher_renders_qwen30_long_context_topology() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "dynamic",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="long-context-contract-test",
        ATTEMPT_ID="attempt-1",
        QWEN30_RECIPE=(
            "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-40K.yaml"
        ),
        QWEN30_NODES="8",
        NUM_PROMPTS_PER_STEP="16",
        NUM_GENERATIONS_PER_PROMPT="16",
        TRAIN_GLOBAL_BATCH_SIZE="256",
        MAX_TOTAL_SEQUENCE_LENGTH="40960",
        MAX_NEW_TOKENS="32768",
        DYNAMIC_SCHEDULE="[[1,2,5],[3,4,4],[5,8,3],[9,16,1],[17,512,0]]",
        NRL_IGNORE_TP_ACCURACY_CHECK="1",
    )

    assert "grpo-qwen3-30ba3b-4n8g-40K.yaml" in output
    assert "grpo.num_prompts_per_step=16" in output
    assert "grpo.num_generations_per_prompt=16" in output
    assert "policy.train_global_batch_size=256" in output
    assert "policy.max_total_sequence_length=40960" in output
    assert "policy.generation.max_new_tokens=32768" in output
    assert "cluster.gpus_per_node=4" in output
    assert "cluster.num_nodes=8" in output
    assert "cluster.segment_size=8" in output
    assert "--nodes=8" in output
    assert "--segment=8" in output
    assert "--gres=gpu:4" in output
    assert "NRL_IGNORE_TP_ACCURACY_CHECK=1" in output


def test_dynamicsd_launcher_renders_fixed_eagle3() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "eagle3_k5")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "num_speculative_tokens_per_batch_size" not in output
    assert "Qwen3-30B-A3B-Thinking-2507-speculator.eagle3" in output


def test_dynamicsd_launcher_renders_low_k_eagle3_controls() -> None:
    k1_output = _dry_run_dynamicsd("qwen30ba3b", "eagle3_k1")
    k2_output = _dry_run_dynamicsd("qwen30ba3b", "eagle3_k2")
    k3_output = _dry_run_dynamicsd("qwen30ba3b", "eagle3_k3")

    assert "speculative_config.num_speculative_tokens=1" in k1_output
    assert "speculative_config.num_speculative_tokens=2" in k2_output
    assert "speculative_config.num_speculative_tokens=3" in k3_output
    assert "num_speculative_tokens_per_batch_size" not in k1_output
    assert "num_speculative_tokens_per_batch_size" not in k2_output
    assert "num_speculative_tokens_per_batch_size" not in k3_output


def test_dynamicsd_launcher_renders_explicit_cudagraph_capture_limit() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "eagle3_k5",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="capture-coverage-test",
        ATTEMPT_ID="attempt-1",
        MAX_CUDAGRAPH_CAPTURE_SIZE="768",
    )

    assert "compilation_config.max_cudagraph_capture_size=768" in output


def test_dynamicsd_launcher_honors_explicit_cudagraph_mode() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "eagle3_k5",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="graph-mode-contract-test",
        ATTEMPT_ID="attempt-1",
        CUDAGRAPH_MODE="PIECEWISE",
    )

    assert "compilation_config.cudagraph_mode=PIECEWISE" in output
    assert "compilation_config.cudagraph_mode=FULL_AND_PIECEWISE" not in output


def test_dynamicsd_launcher_renders_explicit_cudagraph_capture_sizes() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen235b",
        "eagle3_k5",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="capture-shapes-test",
        ATTEMPT_ID="attempt-1",
        MAX_CUDAGRAPH_CAPTURE_SIZE="384",
        CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,32,64,128,192,256,320,384]",
    )

    assert "compilation_config.max_cudagraph_capture_size=384" in output
    assert (
        "compilation_config.cudagraph_capture_sizes="
        "\\[1\\,2\\,4\\,8\\,16\\,32\\,64\\,128\\,192\\,256\\,320\\,384\\]"
    ) in output


def test_dynamicsd_launcher_enables_cudagraph_dispatch_metrics() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "pard_k5",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="cudagraph-dispatch-metrics-test",
        ATTEMPT_ID="attempt-1",
        CUDAGRAPH_DISPATCH_METRICS="true",
    )

    assert "vllm_cfg.env_vars.NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS=true" in output
    assert "++policy.generation.vllm_kwargs.cudagraph_metrics=true" in output
    assert "observability_config.cudagraph_metrics" not in output


def test_dynamicsd_launcher_renders_matched_scheduler_limits() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "baseline",
        REPO_DIR="/lustre/users/sna/RL",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="scheduler-contract-test",
        ATTEMPT_ID="attempt-1",
        MAX_NUM_BATCHED_TOKENS="32768",
        MAX_NUM_SEQS="128",
    )

    assert "policy.generation.vllm_kwargs.max_num_batched_tokens=32768" in output
    assert "policy.generation.vllm_kwargs.max_num_seqs=128" in output


def test_dynamicsd_launcher_reserves_context_without_extending_rl_output() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "eagle3_k5",
        REPO_DIR="/lustre/users/sna/RL",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="context-headroom-test",
        ATTEMPT_ID="attempt-1",
        OUTPUT_MAX_MODEL_LEN="4096",
        SPECDEC_CONTEXT_HEADROOM_TOKENS="32",
    )

    assert "policy.generation._output_max_model_len=4096" in output
    assert "policy.generation.vllm_cfg.max_model_len=4128" in output


def test_dynamicsd_launcher_rejects_invalid_cudagraph_capture_limit() -> None:
    result = _run_script_unchecked(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "eagle3_k5",
        MAX_CUDAGRAPH_CAPTURE_SIZE="not-an-integer",
    )

    assert result.returncode != 0
    assert "MAX_CUDAGRAPH_CAPTURE_SIZE must be a positive integer" in result.stderr


def test_dynamicsd_launcher_renders_aggressive_fixed_k_values() -> None:
    k7_output = _dry_run_dynamicsd("qwen32b", "eagle3_k7")
    k9_output = _dry_run_dynamicsd("qwen32b", "eagle3_k9")

    assert "speculative_config.num_speculative_tokens=7" in k7_output
    assert "speculative_config.num_speculative_tokens=9" in k9_output
    assert "num_speculative_tokens_per_batch_size" not in k7_output
    assert "num_speculative_tokens_per_batch_size" not in k9_output


def test_dynamicsd_launcher_all_includes_k7_and_k9() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "all",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )

    assert "contract-test-attempt-1-qwen30ba3b-eagle3_k7" in output
    assert "contract-test-attempt-1-qwen30ba3b-eagle3_k9" in output


def test_dynamicsd_launcher_qwen30_comparison_includes_all_methods() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "compare",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )

    for variant in (
        "baseline",
        "eagle3_k5",
        "eagle3_k7",
        "eagle3_k9",
        "suffix_k32",
        "pard_k5",
        "pard_k16",
        "dflash_k15",
    ):
        assert f"contract-test-attempt-1-qwen30ba3b-{variant}" in output
    commands = [
        line for line in output.splitlines() if line.startswith("[DRY-RUN] command ")
    ]
    assert len(commands) == 8
    assert all(
        "policy.generation.vllm_kwargs.max_num_batched_tokens=32768" in command
        for command in commands
    )


def test_dynamicsd_launcher_renders_suffix_k32() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "suffix_k32",
        REPO_DIR="/lustre/users/sna/RL",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        ARCTIC_OVERLAY="/lustre/users/sna/arctic-inference-0.1.1",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )

    assert "speculative_config.method=suffix" in output
    assert "speculative_config.num_speculative_tokens=32" in output
    assert "speculative_config.model=" not in output
    assert "speculative_config.draft_tensor_parallel_size" not in output
    assert (
        "PYTHONPATH=/lustre/users/sna/arctic-inference-0.1.1:/lustre/users/sna/RL"
        in output
    )


@pytest.mark.parametrize("variant", ["eagle3_k5", "pard_k5", "dflash_k15"])
def test_dynamicsd_launcher_renders_probabilistic_sampling_for_model_based_specdec(
    variant: str,
) -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", variant)

    assert "speculative_config.rejection_sample_method=standard" in output
    assert "speculative_config.draft_sample_method=probabilistic" in output


def test_dynamicsd_launcher_renders_only_rejection_sampling_for_suffix() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "suffix_k32")

    assert "speculative_config.rejection_sample_method=standard" in output
    assert "speculative_config.draft_sample_method=" not in output


def test_dynamicsd_launcher_keeps_baseline_free_of_sampling_overrides() -> None:
    output = _dry_run_dynamicsd("qwen32b", "baseline")

    assert "speculative_config" not in output


@pytest.mark.parametrize(
    ("environment_name", "environment_value", "expected_message"),
    [
        (
            "REJECTION_SAMPLE_METHOD",
            "random",
            "REJECTION_SAMPLE_METHOD must be standard",
        ),
        (
            "DRAFT_SAMPLE_METHOD",
            "beam",
            "DRAFT_SAMPLE_METHOD must be greedy or probabilistic",
        ),
    ],
)
def test_dynamicsd_launcher_rejects_invalid_sampling_methods(
    environment_name: str,
    environment_value: str,
    expected_message: str,
) -> None:
    result = _run_script_unchecked(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen30ba3b",
        "baseline",
        **{environment_name: environment_value},
    )

    assert result.returncode != 0
    assert expected_message in result.stderr


def test_dynamicsd_launcher_submit_writes_a_consistent_sampling_manifest(
    tmp_path: Path,
) -> None:
    environment, manifest = _prepare_submit_environment(tmp_path)
    environment.update(
        {
            "NUM_PROMPTS_PER_STEP": "16",
            "NUM_GENERATIONS_PER_PROMPT": "16",
            "TRAIN_GLOBAL_BATCH_SIZE": "256",
            "MAX_TOTAL_SEQUENCE_LENGTH": "40960",
            "MAX_NEW_TOKENS": "32768",
        }
    )

    result = _run_script_unchecked(
        DYNAMICSD_LAUNCHER,
        "submit",
        "qwen30ba3b",
        "eagle3_k5",
        **environment,
    )

    assert result.returncode == 0, result.stderr
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].split("\t") == SUBMISSIONS_HEADER.split("\t")
    assert all(len(line.split("\t")) == 30 for line in lines)
    assert lines[1].split("\t")[16:18] == ["standard", "probabilistic"]
    manifest_row = dict(
        zip(
            SUBMISSIONS_HEADER.split("\t"),
            lines[1].split("\t"),
            strict=True,
        )
    )
    assert manifest_row["num_prompts_per_step"] == "16"
    assert manifest_row["num_generations_per_prompt"] == "16"
    assert manifest_row["train_global_batch_size"] == "256"
    assert manifest_row["max_total_sequence_length"] == "40960"
    assert manifest_row["max_new_tokens"] == "32768"


def test_dynamicsd_launcher_submit_rejects_a_legacy_manifest_header(
    tmp_path: Path,
) -> None:
    environment, manifest = _prepare_submit_environment(tmp_path)
    manifest.parent.mkdir(parents=True)
    legacy_header = (
        "timestamp\tmodel\tvariant\tjob_id\tnodes\tsegment\tcommit\t"
        "wandb_run_id\twandb_url\trecipe\tdraft_model\tcontainer\t"
        "container_sha256\tmax_steps\tstatic_k\tdynamic_schedule\tcommand\n"
    )
    manifest.write_text(legacy_header, encoding="utf-8")

    result = _run_script_unchecked(
        DYNAMICSD_LAUNCHER,
        "submit",
        "qwen30ba3b",
        "eagle3_k5",
        **environment,
    )

    assert result.returncode == 2
    assert "submissions manifest header mismatch" in result.stderr
    assert manifest.read_text(encoding="utf-8") == legacy_header


def test_dynamicsd_launcher_renders_pard_with_graph_patch() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "pard_k16")

    assert "speculative_config.method=draft_model" in output
    assert "speculative_config.num_speculative_tokens=16" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "speculative_config.parallel_drafting=true" in output
    assert "models--amd--PARD-Qwen3-0.6B" in output
    assert "NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH=true" in output
    assert "policy.generation.vllm_kwargs.max_num_batched_tokens=32768" in output


def test_dynamicsd_launcher_renders_qwen32_pard_with_target_recipe_topology() -> None:
    output = _dry_run_dynamicsd("qwen32b", "pard_k5")

    assert "grpo-qwen3-32b-4n4g.yaml" in output
    assert "speculative_config.method=draft_model" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "speculative_config.draft_tensor_parallel_size=2" in output
    assert "models--amd--PARD-Qwen3-0.6B" in output
    assert "cluster.segment_size=4" in output


def test_dynamicsd_launcher_renders_qwen30_dflash_k15() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "dflash_k15")

    assert "speculative_config.method=dflash" in output
    assert "speculative_config.num_speculative_tokens=15" in output
    assert "speculative_config.draft_tensor_parallel_size=1" in output
    assert "speculative_config.attention_backend=FLASH_ATTN" in output
    assert "models--inference-optimization--Qwen3-30B-A3B-speculator.dflash" in output
    assert "snapshots/RESOLVED_FROM_REFS_MAIN" in output
    assert "speculative_config.rejection_sample_method=standard" in output
    assert "speculative_config.draft_sample_method=probabilistic" in output


@pytest.mark.parametrize("model", ["qwen32b", "qwen235b", "all"])
def test_dynamicsd_launcher_rejects_dflash_without_exact_checkpoint(
    model: str,
) -> None:
    result = _run_script_unchecked(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        model,
        "dflash_k15",
    )

    assert result.returncode == 2
    assert "dflash_k15 only supports qwen30ba3b" in result.stderr
    assert "qwen30ba3b-dflash_k15" not in result.stdout


def test_dynamicsd_launcher_renders_qwen235b_performance_topology() -> None:
    output = _dry_run_dynamicsd("qwen235b", "eagle3_k9")

    assert "grpo-qwen3-235b-16n4g.yaml" in output
    assert "Qwen3-235B-A22B-Eagle3" in output
    assert "speculative_config.num_speculative_tokens=9" in output
    assert "cluster.segment_size=16" in output
    assert "--nodes=16" in output
    assert "--segment=16" in output
    assert "--gres=gpu:4" in output
    assert "VLLM_PORT=" not in output
    assert "NRL_DISABLE_VLLM_PORT_OVERRIDE=1" in output


def test_dynamicsd_launcher_renders_dynamic_schedule() -> None:
    output = _dry_run_dynamicsd("qwen32b", "dynamic")

    assert "speculative_config.method=eagle3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert (
        "speculative_config.num_speculative_tokens_per_batch_size="
        "\\[\\[1\\,16\\,5\\]\\,\\[17\\,32\\,4\\]\\,"
        "\\[33\\,64\\,3\\]\\,\\[65\\,128\\,1\\]\\,"
        "\\[129\\,512\\,0\\]\\]"
    ) in output


def test_dynamicsd_launcher_keeps_baseline_free_of_specdec() -> None:
    output = _dry_run_dynamicsd("qwen32b", "baseline")

    assert "compilation_config.cudagraph_mode=FULL_AND_PIECEWISE" in output
    assert "speculative_config" not in output
    assert "NRL_VLLM_ENABLE_V2_DRAFT_DECODE_CAPTURE_PROFILE" not in output


def test_long_output_launcher_renders_matched_16k_and_32k_matrix() -> None:
    output = _run_script(
        LONG_OUTPUT_LAUNCHER,
        "dry-run",
        REPO_DIR="/lustre/users/sna/RL",
        LYRIS_ROOT="/lustre/users/sna",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="long-output-contract-test",
        ATTEMPT_ID="attempt-1",
    )

    assert output.count("[LONG-OUTPUT]") == 8
    assert output.count("grpo.num_prompts_per_step=16") == 16
    assert output.count("grpo.num_generations_per_prompt=16") == 16
    assert output.count("policy.train_global_batch_size=256") == 16
    assert output.count("policy.generation.max_new_tokens=16384") == 8
    assert output.count("policy.max_total_sequence_length=20480") == 8
    assert output.count("policy.generation.max_new_tokens=32768") == 8
    assert output.count("policy.max_total_sequence_length=40960") == 8
    assert output.count("policy.generation._output_max_model_len=20480") == 8
    assert output.count("policy.generation.vllm_cfg.max_model_len=20488") == 8
    assert output.count("policy.generation._output_max_model_len=40960") == 8
    assert output.count("policy.generation.vllm_cfg.max_model_len=40960") == 8
    assert "policy.generation.vllm_cfg.max_model_len=40968" not in output
    assert output.count("policy.megatron_cfg.activation_checkpointing=true") == 16
    assert output.count("policy.logprob_batch_size=1") == 8
    assert output.count("policy.generation.vllm_kwargs.max_num_batched_tokens=32768") == 16
    assert output.count("speculative_config.num_speculative_tokens=3") == 8
    assert output.count("speculative_config.model=") == 8
    assert output.count("compilation_config.cudagraph_mode=FULL_AND_PIECEWISE") == 16


def test_long_output_launcher_can_select_the_32k_retry_slice() -> None:
    output = _run_script(
        LONG_OUTPUT_LAUNCHER,
        "dry-run",
        REPO_DIR="/lustre/users/sna/RL",
        LYRIS_ROOT="/lustre/users/sna",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="long-output-32k-retry-contract-test",
        ATTEMPT_ID="attempt-2",
        OUTPUT_LENGTH_SELECTION="32k",
    )

    assert output.count("[LONG-OUTPUT]") == 4
    assert "policy.generation.max_new_tokens=16384" not in output
    assert output.count("policy.generation.max_new_tokens=32768") == 8
    assert output.count("policy.logprob_batch_size=1") == 8


def test_long_output_standard_profile_preserves_caller_qwen30_overrides() -> None:
    output = _run_script(
        LONG_OUTPUT_LAUNCHER,
        "dry-run",
        REPO_DIR="/lustre/users/sna/RL",
        LYRIS_ROOT="/lustre/users/sna",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="long-output-override-contract-test",
        ATTEMPT_ID="attempt-override",
        MODEL_SELECTION="qwen30ba3b",
        OUTPUT_LENGTH_SELECTION="16k",
        VARIANT_SELECTION="k5-control",
        POLICY_MODEL_NAME="/models/custom-target/snapshots/target-revision",
        QWEN30_DRAFT_MODEL="/models/custom-draft/snapshots/draft-revision",
        QWEN30_RECIPE="examples/configs/custom-qwen30.yaml",
        QWEN30_NODES="7",
    )

    assert "policy.model_name=/models/custom-target/snapshots/target-revision" in output
    assert "models/custom-draft/snapshots/draft-revision" in output
    assert "examples/configs/custom-qwen30.yaml" in output
    assert "--nodes=7" in output
    assert "--segment=7" in output


def test_long_output_launcher_can_select_qwen235b_k3_k5_16k_slice() -> None:
    output = _run_script(
        LONG_OUTPUT_LAUNCHER,
        "dry-run",
        REPO_DIR="/lustre/users/sna/RL",
        LYRIS_ROOT="/lustre/users/sna",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="long-output-qwen235b-contract-test",
        ATTEMPT_ID="attempt-3",
        MODEL_SELECTION="qwen235b",
        OUTPUT_LENGTH_SELECTION="16k",
        VARIANT_SELECTION="compare",
    )

    assert output.count("[LONG-OUTPUT]") == 3
    assert "grpo-qwen3-235b-16n4g.yaml" in output
    assert "models--nvidia--Qwen3-235B-A22B-Eagle3" in output
    assert "speculative_config.num_speculative_tokens=3" in output
    assert "speculative_config.num_speculative_tokens=5" in output
    assert "--nodes=16" in output
    assert "--segment=16" in output
    assert "--gres=gpu:4" not in output


def test_long_output_launcher_renders_qwen30_drafter_distribution_matrix() -> None:
    output = _run_script(
        LONG_OUTPUT_LAUNCHER,
        "dry-run",
        REPO_DIR="/lustre/users/sna/RL",
        LYRIS_ROOT="/lustre/users/sna",
        HF_HOME="/lustre/users/sna/hf_home",
        CONTAINER="/lustre/users/sna/nemo-rl.sqsh",
        RUN_TAG="qwen30-drafter-contract-test",
        ATTEMPT_ID="attempt-4",
        MATRIX_SELECTION="qwen30-drafter",
        MAX_STEPS="20",
    )

    rows = [line for line in output.splitlines() if line.startswith("[LONG-OUTPUT]")]
    assert len(rows) == 11
    assert sum("variant=baseline" in row for row in rows) == 3
    assert sum("variant=eagle3_k5" in row for row in rows) == 4
    assert sum("variant=dynamic" in row for row in rows) == 4
    for identity in (
        "base__base",
        "base__instruct2507",
        "instruct2507__instruct2507",
        "thinking2507__thinking2507",
    ):
        assert any(f"identity={identity}" in row for row in rows)
        assert f"/{identity}/qwen30ba3b/" in output
    assert "identity=base__thinking2507" not in output
    assert "[DRAFTER-ALIAS] base=thinking2507" in output

    for target in (
        "models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
    ):
        assert f"policy.model_name=/lustre/users/sna/hf_home/hub/{target}" in output
        assert f"policy.tokenizer.name=/lustre/users/sna/hf_home/hub/{target}" in output

    for drafter in (
        "models--RedHatAI--Qwen3-30B-A3B-speculator.eagle3",
        "models--RedHatAI--Qwen3-30B-A3B-Instruct-2507-speculator.eagle3",
        "models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3",
    ):
        assert drafter in output

    assert output.count("policy.generation.max_new_tokens=16384") == 22
    assert "policy.generation.max_new_tokens=32768" not in output
    assert output.count("speculative_config.num_speculative_tokens=5") == 16
    assert output.count("num_speculative_tokens_per_batch_size") == 8
    assert output.count("compilation_config.cudagraph_mode=PIECEWISE") == 22
    assert "compilation_config.cudagraph_mode=FULL_AND_PIECEWISE" not in output
    assert output.count("compilation_config.max_cudagraph_capture_size=256") == 22
    assert output.count(
        "compilation_config.cudagraph_capture_sizes="
        "\\[1\\,2\\,4\\,8\\,16\\,32\\,64\\,128\\,256\\]"
    ) == 11
    assert "grpo-qwen3-30ba3b-4n8g-40K.yaml" in output
    assert "--nodes=8" in output
    assert "--segment=8" in output
    assert "scheduler_cls=" not in output


def test_qwen30_drafter_final_submit_requires_calibrated_dynamic_schedule() -> None:
    result = _run_script_unchecked(
        LONG_OUTPUT_LAUNCHER,
        "submit",
        MATRIX_SELECTION="qwen30-drafter",
        MAX_STEPS="20",
    )

    assert result.returncode == 2
    assert "DYNAMIC_SCHEDULE is required for a 20-step" in result.stderr


def test_qwen30_drafter_profile_preflights_target_snapshot(tmp_path: Path) -> None:
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    result = _run_script_unchecked(
        LONG_OUTPUT_LAUNCHER,
        "test-only",
        REPO_DIR=str(REPO_ROOT),
        LYRIS_ROOT=str(tmp_path),
        HF_HOME=str(tmp_path / "missing-hf-home"),
        CONTAINER=str(container),
        MATRIX_SELECTION="qwen30-drafter",
        MAX_STEPS="3",
    )

    assert result.returncode == 2
    assert "target model directory not found" in result.stderr


def test_dynamicsd_launcher_defaults_to_aws_dynamically_staged_assets() -> None:
    output = _run_script(
        DYNAMICSD_LAUNCHER,
        "dry-run",
        "qwen32b",
        "dynamic",
        RUN_TAG="contract-test",
        ATTEMPT_ID="attempt-1",
    )

    aws_root = "/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna"
    assert f"CONTAINER={aws_root}/containers/nemo_rl_nightly.sqsh" in output
    assert f"HF_HOME={aws_root}/hf_home" in output
    assert (
        "[DRY-RUN] wandb https://wandb.ai/nvidia/"
        "nemorl-vllm024-dynamicsd-aws-dfw/runs/"
        "contract-test-attempt-1-qwen32b-dynamic"
    ) in output


def test_dynamicsd_launcher_delegates_engine_ports_to_nemorl() -> None:
    output = _dry_run_dynamicsd("qwen30ba3b", "baseline")

    assert "VLLM_PORT=" not in output


def test_dynamicsd_launcher_records_reproducibility_metadata() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    for field in (
        "container",
        "container_sha256",
        "max_steps",
        "static_k",
        "dynamic_schedule",
        "max_cudagraph_capture_size",
        "cudagraph_capture_sizes",
        "command",
    ):
        assert field in source


def test_dynamicsd_launcher_validates_selected_recipe_and_batch_geometry() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    assert 'git -C "${REPO_DIR}" ls-files --error-unmatch "${recipe}"' in source
    assert "total_trajectories % TRAIN_GLOBAL_BATCH_SIZE" in source
    assert "TRAIN_GLOBAL_BATCH_SIZE > total_trajectories" in source


def test_dynamicsd_launcher_rejects_partial_batch_geometry() -> None:
    result = subprocess.run(
        [
            "bash",
            str(DYNAMICSD_LAUNCHER),
            "dry-run",
            "qwen30ba3b",
            "baseline",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "TRAIN_GLOBAL_BATCH_SIZE": "256"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "require positive NUM_PROMPTS_PER_STEP" in result.stderr


def test_dynamicsd_launcher_ignores_generated_submodule_dirt() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    assert source.count("--ignore-submodules=dirty") == 2


def test_dynamicsd_launcher_preserves_logical_lustre_checkout_path() -> None:
    source = DYNAMICSD_LAUNCHER.read_text(encoding="utf-8")

    assert 'logical_pwd="$(pwd -L)"' in source
    assert 'repo_prefix="$(git rev-parse --show-prefix)"' in source


def test_generation_parity_launcher_matches_lyris_topology_and_cuda_graph() -> None:
    output = _dry_run_parity("eagle3_k5", "greedy")

    assert "--account=coreai_dlalgo_llm" in output
    assert "--partition=gb200" in output
    assert "--nodes=1" in output
    assert "--segment=1" in output
    assert "--gres" not in output
    assert "--target-tp 2" in output
    assert "--draft-tp 1" in output
    assert "--gpus-per-node 2" in output
    assert "--max-model-len 4096" in output
    assert "--max-num-batched-tokens 16384" in output
    assert "--mode greedy" in output
    assert "--draft-sample-method probabilistic" in output
    assert "--samples-per-prompt 1" in output
    assert "/lustre/users/sna/qwen32-eagle3" in output


def test_generation_parity_launcher_uses_short_ray_socket_root() -> None:
    output = _dry_run_parity("eagle3_k5", "greedy")

    assert "--ray-log-dir /tmp/nrp-eagle3_k5-greedy" in output
    assert "parity-contract-test/eagle3_k5/greedy/ray_logs" not in output


def test_generation_parity_launcher_rebuilds_worker_venv_from_branch_lock() -> None:
    output = _dry_run_parity("eagle3_k5", "greedy")

    assert "NRL_FORCE_REBUILD_VENVS=true" in output
    assert (
        "NEMO_RL_VENV_DIR=/lustre/users/sna/experiments/"
        "vllm024-generation-parity/parity-contract-test/"
        "eagle3_k5/greedy/venvs"
    ) in output


def test_generation_parity_launcher_keeps_baseline_free_of_draft_model() -> None:
    output = _dry_run_parity("baseline", "sampled")

    assert "--mode sampled" in output
    assert "--samples-per-prompt 64" in output
    assert "--draft-model" not in output


def test_generation_parity_launcher_all_renders_four_independent_jobs() -> None:
    output = _dry_run_parity("all", "all")

    for label in (
        "baseline-greedy",
        "baseline-sampled",
        "eagle3_k5-greedy",
        "eagle3_k5-sampled",
    ):
        assert f"parity-contract-test-{label}" in output
