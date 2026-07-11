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

import csv
import json
import math
import shutil
import shlex
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pytest

from experiments.vllm_024_upgrade.summarize_tail_gated_specdec import (
    MINI_REQUIRED_MANIFEST_FIELDS,
)
from experiments.vllm_024_upgrade.validate_mini_sync_grpo_tail_gate import main
from experiments.vllm_024_upgrade.validate_mini_sync_grpo_tail_gate import (
    _render_activation_scatter,
)

TARGET_REVISION = "0123456789abcdef0123456789abcdef01234567"
TARGET_CHECKPOINT = f"/models/target/snapshots/{TARGET_REVISION}"


class _FakeRun:
    def __init__(self, history: list[dict[str, object]], url: str) -> None:
        self._history = history
        self.url = url

    def scan_history(self, *, keys: list[str]) -> Iterable[Mapping[str, object]]:
        del keys
        return self._history


class _FakeApi:
    def __init__(self, histories: dict[str, list[dict[str, object]]]) -> None:
        self._histories = histories
        self.calls: list[str] = []

    def run(self, path: str) -> _FakeRun:
        self.calls.append(path)
        run_id = path.rsplit("/", maxsplit=1)[-1]
        return _FakeRun(self._histories[run_id], f"https://wandb.example/{run_id}")


def _mini_command(variant: str) -> str:
    run_dir = f"qwen32b/{variant}"
    wandb_run_id = f"run-{variant}"
    tokens = [
        "env",
        "VLLM_USE_V2_MODEL_RUNNER=1",
        "NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS=true",
        f"WANDB_RUN_ID={wandb_run_id}",
        "WANDB_RUN_GROUP=mini-test",
        "WANDB_RESUME=never",
        f"NEMO_RL_VENV_DIR=/tmp/venvs/{variant}",
        "NRL_FORCE_REBUILD_VENVS=true",
        "PYTHONPATH=/repo",
        f"TRITON_CACHE_DIR=/tmp/triton/{variant}",
        f"TORCHINDUCTOR_CACHE_DIR=/tmp/inductor/{variant}",
        "uv",
        "run",
        "examples/run_grpo.py",
        "--config",
        "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
        "grpo.max_num_steps=2",
        "grpo.num_prompts_per_step=16",
        "grpo.num_generations_per_prompt=4",
        "checkpointing.enabled=false",
        f"checkpointing.checkpoint_dir={run_dir}/checkpoints",
        f"policy.model_name={TARGET_CHECKPOINT}",
        "policy.train_global_batch_size=64",
        "policy.max_total_sequence_length=1024",
        "policy.generation.max_new_tokens=1024",
        "policy.generation.temperature=1.0",
        "policy.generation.top_p=1.0",
        "++policy.generation._output_max_model_len=1024",
        "policy.generation.vllm_cfg.max_model_len=1056",
        "policy.generation.vllm_cfg.tensor_parallel_size=2",
        "policy.generation.vllm_cfg.expert_parallel_size=1",
        "policy.generation.vllm_cfg.enforce_eager=false",
        "policy.generation.vllm_cfg.enable_vllm_metrics_logger=true",
        "policy.generation.vllm_cfg.vllm_metrics_logger_interval=0.5",
        "++policy.generation.vllm_cfg.env_vars.NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS=true",
        "++policy.generation.vllm_kwargs.max_num_batched_tokens=16384",
        "++policy.generation.vllm_kwargs.max_num_seqs=1024",
        "++policy.generation.vllm_kwargs.moe_backend=triton",
        "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE",
        "cluster.gpus_per_node=4",
        "cluster.num_nodes=4",
        "cluster.segment_size=4",
        "logger.wandb_enabled=true",
        "logger.tensorboard_enabled=false",
        "logger.wandb.project=nemorl-vllm024-tail-gated-mini-sync-grpo-pre-tyche",
        f"logger.wandb.name={wandb_run_id}",
        "++logger.wandb.entity=nvidia",
        f"logger.log_dir={run_dir}/nemo_logs",
    ]
    if variant != "baseline_v2":
        tokens.extend(
            [
                "++policy.generation.vllm_kwargs.speculative_config.method=eagle3",
                "++policy.generation.vllm_kwargs.speculative_config.model=/models/draft",
                "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=5",
                "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1",
                "++policy.generation.vllm_kwargs.speculative_config.rejection_sample_method=standard",
                "++policy.generation.vllm_kwargs.speculative_config.draft_sample_method=probabilistic",
            ]
        )
    if variant == "fastrl_threshold_v2_k5":
        tokens.extend(
            [
                "++policy.generation.vllm_kwargs.scheduler_cls=nemo_rl.models.generation.vllm.tail_gate_scheduler.TailGatedScheduler",
                "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_mode=threshold",
                "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_threshold=4",
                "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_consecutive_checks=10",
                "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_off_mode=advance_only",
            ]
        )
    return " ".join(tokens)


def _mini_launcher_command(variant: str) -> str:
    run_dir = f"qwen32b/{variant}"
    return " ".join(
        [
            "env",
            "CONTAINER=/containers/nemo.sqsh",
            "MOUNTS=/lustre:/lustre",
            "CONTAINER_WORKDIR=/repo",
            shlex.quote(f"COMMAND={_mini_command(variant)}"),
            f"BASE_LOG_DIR={run_dir}",
            "GPUS_PER_NODE=4",
            "HF_HOME=/lustre/hf_home",
            "PYTHONPATH=/repo",
            "PYTHONDONTWRITEBYTECODE=1",
            "RAY_LOG_SYNC_FREQUENCY=60",
            "NRL_RUNTIME_CHECKOUT=/repo",
            "NRL_EXPECTED_RUNTIME_COMMIT=abc123",
            "TMPDIR=/tmp",
            f"TRITON_CACHE_DIR=/tmp/triton/{variant}",
            f"TORCHINDUCTOR_CACHE_DIR=/tmp/inductor/{variant}",
            "sbatch",
            "--parsable",
            "--account=account",
            "--partition=batch_block1",
            "--nodes=4",
            "--ntasks-per-node=1",
            "--exclusive",
            "--time=02:00:00",
            "--segment=4",
            f"--job-name=account-nemorl.tail-gate-qwen32b-{variant}",
            f"--output={run_dir}/slurm-%j.out",
            "--open-mode=append",
            "--comment=metrics",
            "/repo/ray.sub",
        ]
    )


def _metadata(variant: str) -> dict[str, str]:
    gated = variant == "fastrl_threshold_v2_k5"
    job_id = f"job-{variant}"
    run_dir = f"qwen32b/{variant}"
    ray_log_root = f"{run_dir}/{job_id}-logs"
    command = _mini_command(variant)
    launcher_command = _mini_launcher_command(variant)
    values = {
        "timestamp": "2026-07-10T12:00:00Z",
        "model": "qwen32b",
        "variant": variant,
        "gate_mode": "threshold" if gated else "off",
        "k": "0" if variant == "baseline_v2" else "5",
        "threshold": "4" if gated else "",
        "consecutive_checks": "10" if gated else "",
        "roofline_config_sha256": "",
        "cluster": "pre-tyche",
        "runtime": "nemo-rl",
        "runtime_version": "nightly-20260705",
        "runtime_commit": "abc123",
        "vllm_version": "0.24.0",
        "vllm_commit": "ee0da84a",
        "target_tp": "2",
        "draft_tp": "1",
        "dp": "8",
        "ep": "1",
        "temperature": "1.0",
        "top_p": "1.0",
        "max_osl": "1024",
        "max_model_len": "1056",
        "max_sequence_length": "1024",
        "num_prompts": "16",
        "num_generations": "4",
        "train_gbs": "64",
        "max_num_batched_tokens": "16384",
        "max_num_seqs": "1024",
        "recipe": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
        "container": "/containers/nemo.sqsh",
        "container_sha256": "deadbeef",
        "runner": "v2",
        "graph_mode": "FULL_AND_PIECEWISE",
        "sampling": "standard",
        "draft_sample_method": (
            "not_applicable" if variant == "baseline_v2" else "probabilistic"
        ),
        "job_id": job_id,
        "wandb_run_id": f"run-{variant}",
        "wandb_url": "",
        "run_dir": run_dir,
        "slurm_log_path": f"{run_dir}/slurm-{job_id}.out",
        "ray_driver_log_path": f"{ray_log_root}/ray-driver.log",
        "ray_log_dir": f"{ray_log_root}/ray",
        "launcher_command": launcher_command,
        "command": command,
        "checkout_path": "/repo",
        "ray_sub_path": "/repo/ray.sub",
        "target_checkpoint": TARGET_CHECKPOINT,
        "target_checkpoint_revision": TARGET_REVISION,
        "draft_checkpoint": (
            "not_applicable" if variant == "baseline_v2" else "/models/draft"
        ),
        "command_argv_json": json.dumps(shlex.split(command)),
        "launcher_argv_json": json.dumps(shlex.split(launcher_command)),
    }
    assert set(MINI_REQUIRED_MANIFEST_FIELDS).issubset(values)
    return values


def _history(
    metadata: Mapping[str, str],
    *,
    activation_tick: float = 17.0,
    activation_batch: float = 4.0,
    activations: float = 8.0,
    enabled_ratio: float = 0.25,
    advance_only_ratio: float = 0.75,
    k0_steps: float = 75.0,
    k5_steps: float = 25.0,
    num_drafts: float = 100.0,
    num_accepted_tokens: float = 150.0,
    reward: float = 0.4,
    policy_time: float = 30.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in (1, 2):
        row: dict[str, object] = {
            "_step": step,
            "timing/train/total_step_time": 200.0,
            "timing/train/generation": 100.0,
            "performance/tokens_per_sec_per_gpu": 25.0,
            "performance/generation_tokens_per_sec_per_gpu": 50.0,
            "timing/train/policy_training": policy_time,
            "timing/train/policy_and_reference_logprobs": 20.0,
            "train/reward": reward,
            "train/mean_gen_tokens_per_sample": 512.0,
            "train/gen_kl_error": 0.01,
            "train/loss": 0.2,
            "train/vllm/cudagraph_target_graph_call_ratio": 1.0,
        }
        if metadata["variant"] != "baseline_v2":
            row.update(
                {
                    "train/vllm/spec_num_drafts": num_drafts,
                    "train/vllm/spec_num_draft_tokens": 300.0,
                    "train/vllm/spec_num_accepted_tokens": num_accepted_tokens,
                    "train/vllm/spec_acceptance_rate": 0.5,
                    "train/vllm/spec_acceptance_length": 2.5,
                    "train/vllm/cudagraph_draft_prefill_graph_call_ratio": 1.0,
                    "train/vllm/cudagraph_draft_decode_graph_call_ratio": 1.0,
                }
            )
        if metadata["gate_mode"] == "threshold":
            row.update(
                {
                    "train/vllm/tail_gate_decisions": 100.0,
                    "train/vllm/tail_gate_activations": activations,
                    "train/vllm/tail_gate_enabled_step_ratio": enabled_ratio,
                    "train/vllm/tail_gate_advance_only_step_ratio": advance_only_ratio,
                    "train/vllm/tail_gate_activation_tick": activation_tick,
                    "train/vllm/tail_gate_activation_batch": activation_batch,
                    "train/vllm/tail_gate_activation_seq_len": 512.0,
                    "train/vllm/tail_gate_predicted_speedup": 1.1,
                    "train/vllm/tail_gate_k_0_steps": k0_steps,
                    "train/vllm/tail_gate_k_5_steps": k5_steps,
                }
            )
        rows.append(row)
    return rows


def _cohort() -> list[dict[str, str]]:
    return [
        _metadata("baseline_v2"),
        _metadata("always_on_v2_k5"),
        _metadata("fastrl_threshold_v2_k5"),
    ]


def _write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(materialized[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(materialized)
    for row in materialized:
        slurm_log = path.parent / row["slurm_log_path"]
        ray_driver_log = path.parent / row["ray_driver_log_path"]
        ray_worker_log = (
            path.parent / row["ray_log_dir"] / "session_1" / "logs" / "worker-1.err"
        )
        for log_path in (slurm_log, ray_driver_log, ray_worker_log):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "Training completed through policy step 2.\n", encoding="utf-8"
            )
        (ray_driver_log.parent / ".ray_logs_final_sync_complete").touch()
        (ray_driver_log.parent / ".ray_logs_final_sync_status.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "driver_exit_code": 0,
                    "final_sync_complete": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        evidence_dir = ray_driver_log.parent / ".ray_logs_final_sync_evidence"
        evidence_dir.mkdir()
        (evidence_dir / "head").touch()
        for worker in range(3):
            (evidence_dir / f"worker-{worker}").touch()


def _run_validator(
    tmp_path: Path,
    rows: list[dict[str, str]],
    histories: dict[str, list[dict[str, object]]],
) -> tuple[int, Path]:
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    _write_manifest(manifest, rows)
    result = main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir)],
        api=_FakeApi(histories),
    )
    return result, output_dir


def _manifest_path(manifest: Path, row: Mapping[str, str], field: str) -> Path:
    return manifest.parent / row[field]


def _move_override_before_entrypoint(command: str) -> str:
    tokens = shlex.split(command)
    override = "grpo.max_num_steps=2"
    tokens.remove(override)
    tokens.insert(tokens.index("uv"), override)
    return shlex.join(tokens)


def test_mini_validator_exports_main() -> None:
    assert callable(main)


@pytest.mark.parametrize(
    "active_token",
    [
        "$(touch /tmp/command-substitution)",
        "`touch /tmp/backtick`",
        ">/tmp/redirected.log",
        "&&",
        "|",
        ";",
    ],
)
def test_mini_validator_rejects_shell_active_structured_argv_before_wandb_query(
    tmp_path: Path, active_token: str
) -> None:
    rows = _cohort()
    row = rows[0]
    command_argv = json.loads(row["command_argv_json"])
    command_argv.append(active_token)
    row["command_argv_json"] = json.dumps(command_argv)
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini command:baseline_v2:argv"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("argv_field", "token_prefix", "replacement"),
    [
        ("launcher_argv_json", "CONTAINER=", "/containers/other.sqsh"),
        (
            "command_argv_json",
            "++policy.generation.vllm_kwargs.speculative_config.model=",
            "/models/other-draft",
        ),
        ("launcher_argv_json", "CONTAINER_WORKDIR=", "/other/repo"),
        ("launcher_argv_json", "/repo/ray.sub", "/other/repo/ray.sub"),
    ],
)
def test_mini_validator_rejects_structured_provenance_mismatch_before_wandb_query(
    tmp_path: Path,
    argv_field: str,
    token_prefix: str,
    replacement: str,
) -> None:
    rows = _cohort()
    row = rows[1]
    argv = json.loads(row[argv_field])
    for index, token in enumerate(argv):
        if token == token_prefix or token.startswith(token_prefix):
            argv[index] = (
                replacement if token == token_prefix else f"{token_prefix}{replacement}"
            )
            break
    else:
        raise AssertionError(f"token not found: {token_prefix}")
    row[argv_field] = json.dumps(argv)
    if argv_field == "command_argv_json":
        row["command"] = shlex.join(argv)
        launcher_argv = json.loads(row["launcher_argv_json"])
        command_index = next(
            index
            for index, token in enumerate(launcher_argv)
            if token.startswith("COMMAND=")
        )
        launcher_argv[command_index] = f"COMMAND={row['command']}"
        row["launcher_argv_json"] = json.dumps(launcher_argv)
        row["launcher_command"] = shlex.join(launcher_argv)
    else:
        row["launcher_command"] = shlex.join(argv)
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini .*provenance"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == []
    assert not output_dir.exists()


def test_mini_validator_accepts_completed_matched_threshold_smoke(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}

    result, output_dir = _run_validator(tmp_path, rows, histories)

    assert result == 0
    payload = json.loads((output_dir / "mini_summary.json").read_text())
    threshold = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert threshold["status"] == "final"
    assert threshold["mini_health_passed"] is True
    assert threshold["activation_tick"] == 17.0
    assert threshold["tail_gate_k0_steps"] == 75.0
    assert threshold["tail_gate_k5_steps"] == 25.0


def test_mini_validator_rejects_incomplete_matrix_before_wandb_query(
    tmp_path: Path,
) -> None:
    rows = _cohort()[:2]
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="mini manifest variants must be exactly"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize("field", ["wandb_run_id", "job_id"])
def test_mini_validator_rejects_duplicate_run_identifiers_before_wandb_query(
    tmp_path: Path, field: str
) -> None:
    rows = _cohort()
    rows[1][field] = rows[0][field]
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match=rf"duplicate {field}"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("variant", "field", "invalid_value"),
    [
        ("baseline_v2", "gate_mode", "threshold"),
        ("baseline_v2", "k", "5"),
        ("baseline_v2", "threshold", "32"),
        ("baseline_v2", "consecutive_checks", "10"),
        ("always_on_v2_k5", "gate_mode", "threshold"),
        ("always_on_v2_k5", "k", "0"),
        ("always_on_v2_k5", "threshold", "32"),
        ("always_on_v2_k5", "consecutive_checks", "10"),
        ("always_on_v2_k5", "draft_sample_method", "greedy"),
        ("fastrl_threshold_v2_k5", "gate_mode", "off"),
        ("fastrl_threshold_v2_k5", "k", "0"),
        ("fastrl_threshold_v2_k5", "threshold", "0"),
        ("fastrl_threshold_v2_k5", "consecutive_checks", "9"),
        ("fastrl_threshold_v2_k5", "draft_sample_method", "greedy"),
    ],
)
def test_mini_validator_rejects_invalid_variant_mapping_before_wandb_query(
    tmp_path: Path, variant: str, field: str, invalid_value: str
) -> None:
    rows = _cohort()
    target = next(row for row in rows if row["variant"] == variant)
    target[field] = invalid_value
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    error_pattern = (
        "invalid mini threshold"
        if variant == "fastrl_threshold_v2_k5" and field == "threshold"
        else rf"invalid mini manifest field:{variant}:{field}"
    )
    with pytest.raises(ValueError, match=error_pattern):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("model", "qwen30ba3b"),
        ("cluster", "lyris-gb200"),
        ("runtime", "other-runtime"),
        ("recipe", "examples/configs/recipes/llm/performance/other.yaml"),
        ("target_tp", "4"),
        ("draft_tp", "2"),
        ("dp", "4"),
        ("ep", "2"),
        ("temperature", "0.7"),
        ("top_p", "0.9"),
        ("max_osl", "512"),
        ("max_model_len", "2080"),
        ("max_sequence_length", "2048"),
        ("num_prompts", "32"),
        ("num_generations", "8"),
        ("train_gbs", "128"),
        ("max_num_batched_tokens", "4096"),
        ("max_num_seqs", "64"),
        ("runner", "v1"),
        ("graph_mode", "PIECEWISE"),
        ("sampling", "typical"),
        ("command", "run checkpointing.enabled=true"),
        (
            "command",
            "run grpo.max_num_steps=2 checkpointing.enabled=falseish",
        ),
        (
            "command",
            "run grpo.max_num_steps=2 checkpointing.enabled=false "
            "checkpointing.enabled=true",
        ),
        (
            "command",
            "run grpo.max_num_steps=20 checkpointing.enabled=false",
        ),
    ],
)
def test_mini_validator_rejects_non_exact_workload_before_wandb_query(
    tmp_path: Path, field: str, invalid_value: str
) -> None:
    rows = _cohort()
    rows[0][field] = invalid_value
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)
    if field == "runner":
        error_pattern = "runner"
    elif field == "command":
        error_pattern = "invalid mini command"
    elif field in {"dp", "num_prompts", "num_generations"}:
        error_pattern = "invalid mini"
    else:
        error_pattern = rf"invalid mini manifest (field|provenance):.*{field}"

    with pytest.raises(ValueError, match=error_pattern):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("variant", "old", "new"),
    [
        ("baseline_v2", "cluster.num_nodes=4", "cluster.num_nodes=8"),
        ("baseline_v2", "cluster.gpus_per_node=4", "cluster.gpus_per_node=2"),
        ("baseline_v2", "cluster.segment_size=4", "cluster.segment_size=2"),
        (
            "baseline_v2",
            "cluster.segment_size=4",
            "cluster.segment_size=4 --gres=gpu:4",
        ),
        (
            "baseline_v2",
            "VLLM_USE_V2_MODEL_RUNNER=1",
            "VLLM_USE_V2_MODEL_RUNNER=0",
        ),
        (
            "baseline_v2",
            "cudagraph_mode=FULL_AND_PIECEWISE",
            "cudagraph_mode=PIECEWISE",
        ),
        (
            "baseline_v2",
            "policy.generation.temperature=1.0",
            "policy.generation.temperature=0.7",
        ),
        (
            "baseline_v2",
            "policy.generation.top_p=1.0",
            "policy.generation.top_p=0.9",
        ),
        (
            "baseline_v2",
            "--config examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
            "--config examples/configs/recipes/llm/performance/other.yaml",
        ),
        (
            "baseline_v2",
            "examples/run_grpo.py",
            "examples/other.py",
        ),
        (
            "baseline_v2",
            "cluster.segment_size=4",
            "cluster.segment_size=4 "
            "++policy.generation.vllm_kwargs.speculative_config.draft_sample_method=probabilistic",
        ),
        (
            "always_on_v2_k5",
            "rejection_sample_method=standard",
            "rejection_sample_method=typical",
        ),
        (
            "always_on_v2_k5",
            "draft_sample_method=probabilistic",
            "draft_sample_method=greedy",
        ),
        (
            "always_on_v2_k5",
            " ++policy.generation.vllm_kwargs.speculative_config.model=/models/draft",
            "",
        ),
        (
            "fastrl_threshold_v2_k5",
            "sd_tail_gate_threshold=4",
            "sd_tail_gate_threshold=5",
        ),
    ],
)
def test_mini_validator_rejects_command_contract_mismatch_before_wandb_query(
    tmp_path: Path, variant: str, old: str, new: str
) -> None:
    rows = _cohort()
    row = next(row for row in rows if row["variant"] == variant)
    assert old in row["command"]
    row["command"] = row["command"].replace(old, new)
    histories = {item["wandb_run_id"]: _history(item) for item in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match=rf"invalid mini command:{variant}"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda command: command.replace("uv run", "/usr/bin/true", 1),
        lambda command: f"{command} unknown.override=true",
        lambda command: f"{command} grpo.max_num_steps=2",
        _move_override_before_entrypoint,
    ],
    ids=[
        "wrong-executable",
        "unknown-override",
        "duplicate-override",
        "malformed-order",
    ],
)
def test_mini_validator_rejects_command_structure_bypasses_before_wandb_query(
    tmp_path: Path, mutate: Callable[[str], str]
) -> None:
    rows = _cohort()
    row = rows[0]
    row["command"] = mutate(row["command"])
    row["launcher_command"] = _mini_launcher_command(row["variant"]).replace(
        shlex.quote(f"COMMAND={_mini_command(row['variant'])}"),
        shlex.quote(f"COMMAND={row['command']}"),
    )
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini command:baseline_v2"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == []
    assert not output_dir.exists()


def test_mini_validator_requires_nonempty_command_for_every_arm(tmp_path: Path) -> None:
    rows = _cohort()
    rows[1]["command"] = ""
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="missing mini manifest fields:command"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("--nodes=4", "--nodes=8"),
        ("GPUS_PER_NODE=4", "GPUS_PER_NODE=2"),
        ("--segment=4", "--segment=2"),
        ("--segment=4", "--segment=4 --gres=gpu:4"),
        ("RAY_LOG_SYNC_FREQUENCY=60", "RAY_LOG_SYNC_FREQUENCY="),
        ("slurm-%j.out", "different-%j.out"),
        ("checkpointing.enabled=false", "checkpointing.enabled=true"),
    ],
)
def test_mini_validator_rejects_launcher_command_mismatch_before_wandb_query(
    tmp_path: Path, old: str, new: str
) -> None:
    rows = _cohort()
    row = rows[0]
    assert old in row["launcher_command"]
    row["launcher_command"] = row["launcher_command"].replace(old, new)
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini launcher command:baseline_v2"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "suffix",
    [
        " --nodes=8 --nodes=4",
        " --account=other",
        " unexpected-positional",
    ],
    ids=["options-after-ray-sub", "option-after-ray-sub", "positional-after-ray-sub"],
)
def test_mini_validator_rejects_tokens_after_ray_sub_before_wandb_query(
    tmp_path: Path, suffix: str
) -> None:
    rows = _cohort()
    rows[0]["launcher_command"] += suffix
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini launcher command:baseline_v2"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == []
    assert not output_dir.exists()


def test_mini_validator_rejects_duplicate_or_unknown_sbatch_options_before_query(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    rows[0]["launcher_command"] = rows[0]["launcher_command"].replace(
        "--nodes=4", "--nodes=8 --nodes=4 --unknown=value", 1
    )
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini launcher command:baseline_v2"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == []
    assert not output_dir.exists()


def test_mini_validator_rejects_conflicting_sbatch_option_after_ray_sub(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    rows[0]["launcher_command"] = rows[0]["launcher_command"].replace(
        "--nodes=4", "--nodes=8", 1
    )
    rows[0]["launcher_command"] += " --nodes=4"
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini launcher command:baseline_v2"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"num_prompts": "15"}, "global rollouts not divisible by dp"),
        (
            {"num_prompts": "8"},
            "local scheduler capacity:baseline_v2:4:expected:8",
        ),
    ],
)
def test_mini_validator_rejects_invalid_local_scheduler_capacity_before_query(
    tmp_path: Path, updates: dict[str, str], error: str
) -> None:
    rows = _cohort()
    for row in rows:
        row.update(updates)
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match=error):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize("threshold", ["5", "8"])
def test_mini_validator_rejects_non_contract_or_capacity_threshold_before_query(
    tmp_path: Path, threshold: str
) -> None:
    rows = _cohort()
    threshold_row = rows[-1]
    threshold_row["threshold"] = threshold
    threshold_row["command"] = threshold_row["command"].replace(
        "sd_tail_gate_threshold=4", f"sd_tail_gate_threshold={threshold}"
    )
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="threshold"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


def test_mini_validator_uses_manifest_wandb_url_for_each_run(tmp_path: Path) -> None:
    rows = _cohort()
    for row in rows:
        old_command = row["command"]
        row["wandb_url"] = (
            "https://wandb.ai/manifest-entity/manifest-project/runs/"
            f"{row['wandb_run_id']}"
        )
        row["command"] = (
            row["command"]
            .replace(
                "logger.wandb.project=nemorl-vllm024-tail-gated-mini-sync-grpo-pre-tyche",
                "logger.wandb.project=manifest-project",
            )
            .replace(
                "++logger.wandb.entity=nvidia",
                "++logger.wandb.entity=manifest-entity",
            )
        )
        row["launcher_command"] = row["launcher_command"].replace(
            shlex.quote(f"COMMAND={old_command}"),
            shlex.quote(f"COMMAND={row['command']}"),
        )
        row["command_argv_json"] = json.dumps(shlex.split(row["command"]))
        row["launcher_argv_json"] = json.dumps(shlex.split(row["launcher_command"]))
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    result = main(
        [
            "--manifest",
            str(manifest),
            "--entity",
            "wrong-entity",
            "--project",
            "wrong-project",
            "--output-dir",
            str(output_dir),
        ],
        api=api,
    )

    assert result == 0
    assert api.calls == [
        f"manifest-entity/manifest-project/{row['wandb_run_id']}" for row in rows
    ]
    payload = json.loads((output_dir / "mini_summary.json").read_text())
    urls_by_variant = {row["variant"]: row["wandb_url"] for row in payload}
    assert urls_by_variant == {row["variant"]: row["wandb_url"] for row in rows}


def test_mini_validator_rejects_conflicting_checkpoint_provenance(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    rows[0]["command"] = "run checkpointing.enabled=true"
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini command:baseline_v2"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


def test_mini_validator_fallback_matches_mini_launcher_project(tmp_path: Path) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir)],
        api=api,
    )

    assert result == 0
    assert api.calls == [
        "nvidia/nemorl-vllm024-tail-gated-mini-sync-grpo-pre-tyche/"
        f"{row['wandb_run_id']}"
        for row in rows
    ]


def test_mini_validator_accepts_explicit_lyris_cluster(tmp_path: Path) -> None:
    rows = _cohort()
    for row in rows:
        row["cluster"] = "lyris-gb200"
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    result = main(
        [
            "--manifest",
            str(manifest),
            "--expected-cluster",
            "lyris-gb200",
            "--output-dir",
            str(output_dir),
        ],
        api=api,
    )

    assert result == 0


def test_mini_validator_rejects_wandb_url_run_id_mismatch_before_query(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    rows[0]["wandb_url"] = "https://wandb.ai/nvidia/project/runs/different-run"
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="wandb_url run ID mismatch:baseline_v2"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("updates", "failure"),
    [
        ({"activation_tick": 0.0}, "activation_tick"),
        ({"activation_batch": 33.0}, "activation_batch"),
        ({"enabled_ratio": 1.0}, "gate_enabled_ratio"),
        ({"advance_only_ratio": 0.0}, "gate_advance_only_ratio"),
        ({"k0_steps": 0.0}, "tail_gate_k0_steps"),
        ({"k5_steps": 0.0}, "tail_gate_k5_steps"),
        ({"num_drafts": 0.0}, "num_drafts"),
        ({"num_accepted_tokens": 0.0}, "num_accepted_tokens"),
        ({"reward": math.nan}, "reward"),
        ({"policy_time": 0.0}, "policy_training"),
    ],
)
def test_mini_validator_rejects_failed_threshold_health_gate(
    tmp_path: Path, updates: dict[str, float], failure: str
) -> None:
    rows = _cohort()
    histories = {
        row["wandb_run_id"]: _history(
            row,
            **(updates if row["variant"] == "fastrl_threshold_v2_k5" else {}),
        )
        for row in rows
    }

    result, output_dir = _run_validator(tmp_path, rows, histories)

    payload = json.loads((output_dir / "mini_summary.json").read_text())
    threshold = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold["status"] in {"partial", "health_failed"}
    assert failure in threshold["reason"]


def test_mini_validator_rejects_activation_count_above_worker_ranks(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {
        row["wandb_run_id"]: _history(
            row,
            **(
                {"activations": 17.0}
                if row["variant"] == "fastrl_threshold_v2_k5"
                else {}
            ),
        )
        for row in rows
    }

    result, output_dir = _run_validator(tmp_path, rows, histories)

    payload = json.loads((output_dir / "mini_summary.json").read_text())
    threshold = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold["reason"] == (
        "mini_health_failed:tail_gate_activations:expected_range=(0,16]:actual=17.0"
    )


def test_mini_validator_rejects_available_structured_fallback_counter(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    threshold = rows[-1]
    histories[threshold["wandb_run_id"]][0][
        "train/vllm/cudagraph_target_fallback_missing_key"
    ] = 1.0

    result, output_dir = _run_validator(tmp_path, rows, histories)

    payload = json.loads((output_dir / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == "mini_health_failed:cuda_graph_fallback_counter"


@pytest.mark.parametrize(
    ("log_field", "log_text", "failure"),
    [
        (
            "slurm_log_path",
            "slurmstepd: error: Detected 1 oom_kill event in StepId=4242.batch\n",
            "oom",
        ),
        (
            "ray_driver_log_path",
            "torch.OutOfMemoryError: CUDA out of memory\n",
            "oom",
        ),
        (
            "slurm_log_path",
            "slurmstepd: error: OOM detected for task 0\n",
            "oom",
        ),
        (
            "ray_driver_log_path",
            "torch.distributed.DistBackendError: NCCL error: remote process exited\n",
            "nccl",
        ),
        (
            "ray_log_dir",
            "DistBackendError: NCCL communicator timed out during all_reduce\n",
            "nccl",
        ),
        (
            "ray_driver_log_path",
            "RuntimeError: NCCL error: unhandled system error\n",
            "nccl",
        ),
        (
            "ray_driver_log_path",
            "ERROR: q-cache mismatch detected during replay\n",
            "q_cache",
        ),
        (
            "ray_driver_log_path",
            "q-cache mismatch detected during replay\n",
            "q_cache",
        ),
        (
            "ray_log_dir",
            "AssertionError: q_cache must be empty before replay\n",
            "q_cache",
        ),
        (
            "ray_driver_log_path",
            "stale draft IDs observed: 2\n",
            "stale_draft_id",
        ),
        (
            "ray_log_dir",
            "stale draft token IDs detected: [151936]\n",
            "stale_draft_id",
        ),
        (
            "ray_log_dir",
            "stale draft IDs: [151936]\n",
            "stale_draft_id",
        ),
        (
            "ray_driver_log_path",
            "invalid tokens found: 3\n",
            "invalid_token",
        ),
        (
            "ray_log_dir",
            "ValueError: invalid token id 151936\n",
            "invalid_token",
        ),
        (
            "ray_log_dir",
            "invalid tokens: 3\n",
            "invalid_token",
        ),
        (
            "ray_driver_log_path",
            "AssertionError: tokens_left_for_obs=-1 should not be negative\n",
            "tokens_left_for_obs",
        ),
        (
            "ray_driver_log_path",
            "RuntimeError: policy loss is NaN\n",
            "nan",
        ),
        (
            "ray_driver_log_path",
            "WARNING: CUDA graph fallback to eager execution\n",
            "cuda_graph_fallback",
        ),
        (
            "ray_log_dir",
            "ERROR: uncaptured CUDA graph execution\n",
            "cuda_graph_fallback",
        ),
        (
            "ray_driver_log_path",
            "eager_fallback_count=1\n",
            "cuda_graph_fallback",
        ),
        (
            "ray_log_dir",
            "CUDA graph fallback count: 2\n",
            "cuda_graph_fallback",
        ),
        (
            "ray_driver_log_path",
            "RuntimeError: CUDA graph capture failed\n",
            "cuda_graph_fallback",
        ),
        (
            "ray_log_dir",
            "vllm:cudagraph_target_fallback_count=1\n",
            "cuda_graph_fallback",
        ),
        (
            "ray_driver_log_path",
            "[rank3]: RuntimeError: CUDA graph capture failed\n",
            "cuda_graph_fallback",
        ),
        (
            "ray_log_dir",
            "(RayWorker pid=42) IndexError: CUDA graph replay failed\n",
            "cuda_graph_fallback",
        ),
        (
            "ray_log_dir",
            "\x1b[31m(VllmGenerationWorker pid=42) RuntimeError: CUDA graph capture failed\x1b[0m\n",
            "cuda_graph_fallback",
        ),
    ],
)
def test_mini_validator_recursively_rejects_explicit_log_failure_signatures(
    tmp_path: Path, log_field: str, log_text: str, failure: str
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    threshold = rows[-1]
    log_path = _manifest_path(manifest, threshold, log_field)
    if log_field == "ray_log_dir":
        log_path = log_path / "session_1" / "logs" / "worker-1.err"
    log_path.write_text(log_text, encoding="utf-8")

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["status"] == "health_failed"
    assert threshold_row["reason"] == f"mini_health_failed:logs:{failure}"


@pytest.mark.parametrize(
    "log_field", ["slurm_log_path", "ray_driver_log_path", "ray_log_dir"]
)
def test_mini_validator_treats_missing_required_log_as_health_failure(
    tmp_path: Path, log_field: str
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    threshold = rows[-1]
    missing_path = _manifest_path(manifest, threshold, log_field)
    if missing_path.is_dir():
        shutil.rmtree(missing_path)
    else:
        missing_path.unlink()

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == f"mini_health_failed:log_missing:{log_field}"


def test_mini_validator_requires_at_least_one_recursive_ray_text_log(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    ray_log_dir = _manifest_path(manifest, rows[-1], "ray_log_dir")
    shutil.rmtree(ray_log_dir)
    ray_log_dir.mkdir()

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == "mini_health_failed:log_empty:ray_log_dir"


def test_mini_validator_scans_every_restart_attempt(tmp_path: Path) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    threshold = rows[-1]
    run_dir = manifest.parent / threshold["run_dir"]
    restart_dir = run_dir / f"{threshold['job_id']}-1-logs"
    restart_driver = restart_dir / "ray-driver.log"
    restart_worker = restart_dir / "ray/session_2/logs/worker-2.err"
    for log_path in (restart_driver, restart_worker):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("Training completed through policy step 2.\n")
    (restart_dir / ".ray_logs_final_sync_complete").touch()
    (restart_dir / ".ray_logs_final_sync_status.json").write_text(
        '{"schema_version":1,"driver_exit_code":0,"final_sync_complete":true}\n'
    )
    evidence_dir = restart_dir / ".ray_logs_final_sync_evidence"
    evidence_dir.mkdir()
    (evidence_dir / "head").touch()
    for worker in range(3):
        (evidence_dir / f"worker-{worker}").touch()
    _manifest_path(manifest, threshold, "ray_driver_log_path").write_text(
        "RuntimeError: NCCL error: unhandled system error\n", encoding="utf-8"
    )

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == "mini_health_failed:logs:nccl"


def test_mini_validator_requires_completion_marker_on_final_restart(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    threshold = rows[-1]
    run_dir = manifest.parent / threshold["run_dir"]
    restart_dir = run_dir / f"{threshold['job_id']}-2-logs"
    restart_driver = restart_dir / "ray-driver.log"
    restart_worker = restart_dir / "ray/session_3/logs/worker-3.err"
    for log_path in (restart_driver, restart_worker):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("Training completed through policy step 2.\n")

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == (
        "mini_health_failed:log_missing:final_sync_marker"
    )


def test_mini_validator_rejects_failed_driver_despite_completion_marker(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    threshold = rows[-1]
    status_path = (
        manifest.parent / threshold["ray_driver_log_path"]
    ).parent / ".ray_logs_final_sync_status.json"
    status_path.write_text(
        '{"schema_version":1,"driver_exit_code":17,"final_sync_complete":true}\n',
        encoding="utf-8",
    )

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == "mini_health_failed:driver_exit_code:17"


def test_mini_validator_requires_per_node_final_sync_evidence(tmp_path: Path) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    threshold = rows[-1]
    evidence_dir = (
        manifest.parent / threshold["ray_driver_log_path"]
    ).parent / ".ray_logs_final_sync_evidence"
    (evidence_dir / "worker-2").unlink()

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == (
        "mini_health_failed:log_missing:final_sync_node_evidence"
    )


@pytest.mark.parametrize(
    ("replaced_name", "replacement_name", "replacement_is_directory"),
    [
        ("worker-2", "unexpected", False),
        ("worker-2", "worker-2", True),
    ],
)
def test_mini_validator_requires_exact_regular_final_sync_evidence(
    tmp_path: Path,
    replaced_name: str,
    replacement_name: str,
    replacement_is_directory: bool,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    threshold = rows[-1]
    evidence_dir = (
        manifest.parent / threshold["ray_driver_log_path"]
    ).parent / ".ray_logs_final_sync_evidence"
    (evidence_dir / replaced_name).unlink()
    replacement = evidence_dir / replacement_name
    if replacement_is_directory:
        replacement.mkdir()
    else:
        replacement.touch()

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == (
        "mini_health_failed:log_missing:final_sync_node_evidence"
    )


def test_mini_validator_requires_final_attempt_driver_and_ray_evidence(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    threshold = rows[-1]
    run_dir = manifest.parent / threshold["run_dir"]
    restart_dir = run_dir / f"{threshold['job_id']}-3-logs"
    restart_dir.mkdir(parents=True)
    (restart_dir / ".ray_logs_final_sync_complete").touch()
    (restart_dir / ".ray_logs_final_sync_status.json").write_text(
        '{"schema_version":1,"driver_exit_code":0,"final_sync_complete":true}\n',
        encoding="utf-8",
    )

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    payload = json.loads((tmp_path / "output" / "mini_summary.json").read_text())
    threshold_row = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold_row["reason"] == (
        "mini_health_failed:log_missing:ray_driver_log_path"
    )


def test_mini_validator_ignores_benign_log_mentions(tmp_path: Path) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    benign = (
        "OOM avoidance enabled; NaN checks configured.\n"
        "NCCL timeout is configured for 600 seconds.\n"
        "CUDA graph fallback count: 0; eager_fallback_count=0.\n"
        "CUDA graph fallback was disabled.\n"
        "No uncaptured CUDA graphs were observed.\n"
        "q-cache mismatch check passed.\n"
        "stale draft IDs observed: 0.\n"
        "invalid tokens found: 0.\n"
        "tokens_left_for_obs=128; tokens_left_for_obs=-0.\n"
    )
    for row in rows:
        _manifest_path(manifest, row, "slurm_log_path").write_text(
            benign, encoding="utf-8"
        )
        _manifest_path(manifest, row, "ray_driver_log_path").write_text(
            benign, encoding="utf-8"
        )
        (
            _manifest_path(manifest, row, "ray_log_dir")
            / "session_1"
            / "logs"
            / "worker-1.err"
        ).write_text(benign, encoding="utf-8")

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "output")],
        api=_FakeApi(histories),
    )

    assert result == 0


def test_mini_validator_rejects_non_ray_sub_log_layout_before_wandb_query(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    rows[-1]["ray_driver_log_path"] = "recorded/threshold-driver.log"
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi({})
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="invalid mini log provenance"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == []
    assert not output_dir.exists()


def test_mini_validator_reuses_exact_collector_cohort_matching(tmp_path: Path) -> None:
    rows = _cohort()
    rows[1]["container_sha256"] = "different"
    histories = {row["wandb_run_id"]: _history(row) for row in rows}

    result, output_dir = _run_validator(tmp_path, rows, histories)

    payload = json.loads((output_dir / "mini_summary.json").read_text())
    assert result == 1
    assert all(row["status"] == "partial" for row in payload)
    reasons = {row["reason"] for row in payload}
    assert all(
        "comparison_failed:missing matched always-on" in reason for reason in reasons
    )


def test_activation_scatter_is_deterministic_and_identifies_the_event(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    first_result, first_output = _run_validator(tmp_path / "first", rows, histories)
    second_result, second_output = _run_validator(
        tmp_path / "second", list(reversed(rows)), histories
    )

    assert first_result == second_result == 0
    first = (first_output / "tail_gate_activation_events.html").read_bytes()
    second = (second_output / "tail_gate_activation_events.html").read_bytes()
    assert first == second
    report = first.decode()
    assert "Scheduler tick" in report
    assert "Inflight batch" in report
    assert "threshold=4" in report
    assert "OFF-to-ON" in report
    assert "tick=17" in report
    assert "batch=4" in report
    assert "stable speedup" not in report
    assert "two-step smoke makes no speedup claim" in report


def test_activation_scatter_escapes_event_labels() -> None:
    report = _render_activation_scatter(
        [
            {
                "variant": "<script>alert(1)</script>",
                "job_id": "job",
                "step": 1,
                "tick": 2.0,
                "batch": 1.0,
            }
        ],
        threshold=4,
    )

    assert "<script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
