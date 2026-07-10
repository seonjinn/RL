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

"""Validate the two-step synchronous GRPO tail-gate smoke."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import re
import shlex
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, cast
from urllib.parse import unquote, urlparse

from experiments.vllm_024_upgrade.summarize_tail_gated_specdec import (
    MINI_REQUIRED_MANIFEST_FIELDS,
    REQUIRED_ROW_FIELDS,
    ComparisonRow,
    RunSummary,
    WandbApi,
    _claim_output_directory,
    _empty_comparison_row,
    _empty_summary,
    _history_keys,
    _is_finite_number,
    _read_manifest,
    _validate_manifest_rows,
    _write_atomic,
    _write_csv,
    build_comparison_rows,
    summarize_history,
)


MINI_STEPS = {1, 2}
MINI_EXPECTED_NODES = 4
MINI_EXPECTED_GLOBAL_ROLLOUTS = 64
MINI_EXPECTED_LOCAL_CAPACITY = 8
MINI_EXPECTED_THRESHOLD = 4
DEFAULT_WANDB_ENTITY = "nvidia"
DEFAULT_WANDB_PROJECT = "nemorl-vllm024-tail-gated-mini-sync-grpo-pre-tyche"
REQUIRED_COMMON_CONFIG = {
    "model": "qwen32b",
    "cluster": "pre-tyche",
    "runtime": "nemo-rl",
    "recipe": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
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
    "runner": "v2",
    "graph_mode": "FULL_AND_PIECEWISE",
    "sampling": "standard",
}
REQUIRED_VARIANT_CONFIG = {
    "baseline_v2": {
        "gate_mode": "off",
        "k": "0",
        "threshold": "",
        "consecutive_checks": "",
        "draft_sample_method": "not_applicable",
    },
    "always_on_v2_k5": {
        "gate_mode": "off",
        "k": "5",
        "threshold": "",
        "consecutive_checks": "",
        "draft_sample_method": "probabilistic",
    },
    "fastrl_threshold_v2_k5": {
        "gate_mode": "threshold",
        "k": "5",
        "consecutive_checks": "10",
        "draft_sample_method": "probabilistic",
    },
}
MINI_METRIC_KEYS = {
    "tail_gate_k0_steps": "train/vllm/tail_gate_k_0_steps",
    "tail_gate_k5_steps": "train/vllm/tail_gate_k_5_steps",
}
MINI_ROW_FIELDS = (
    *REQUIRED_ROW_FIELDS,
    "mini_health_passed",
    *MINI_METRIC_KEYS,
)
MINI_COMMAND_ENV_ASSIGNMENTS: dict[str, str | None] = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS": "true",
    "WANDB_RUN_ID": None,
    "WANDB_RUN_GROUP": None,
    "WANDB_RESUME": "never",
    "NEMO_RL_VENV_DIR": None,
    "NRL_FORCE_REBUILD_VENVS": "true",
    "PYTHONPATH": None,
    "TRITON_CACHE_DIR": None,
    "TORCHINDUCTOR_CACHE_DIR": None,
}
MINI_COMMAND_ASSIGNMENTS = {
    "grpo.max_num_steps": "2",
    "grpo.num_prompts_per_step": "16",
    "grpo.num_generations_per_prompt": "4",
    "checkpointing.enabled": "false",
    "policy.train_global_batch_size": "64",
    "policy.max_total_sequence_length": "1024",
    "policy.generation.max_new_tokens": "1024",
    "policy.generation.temperature": "1.0",
    "policy.generation.top_p": "1.0",
    "policy.generation._output_max_model_len": "1024",
    "policy.generation.vllm_cfg.max_model_len": "1056",
    "policy.generation.vllm_cfg.tensor_parallel_size": "2",
    "policy.generation.vllm_cfg.expert_parallel_size": "1",
    "policy.generation.vllm_cfg.enforce_eager": "false",
    "policy.generation.vllm_cfg.enable_vllm_metrics_logger": "true",
    "policy.generation.vllm_cfg.vllm_metrics_logger_interval": "0.5",
    "++policy.generation.vllm_cfg.env_vars.NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS": (  # noqa: E501
        "true"
    ),
    "++policy.generation.vllm_kwargs.max_num_batched_tokens": "16384",
    "++policy.generation.vllm_kwargs.max_num_seqs": "1024",
    "++policy.generation.vllm_kwargs.moe_backend": "triton",
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode": (
        "FULL_AND_PIECEWISE"
    ),
    "cluster.gpus_per_node": "4",
    "cluster.num_nodes": "4",
    "cluster.segment_size": "4",
    "logger.wandb_enabled": "true",
    "logger.tensorboard_enabled": "false",
    "logger.wandb.project": None,
    "logger.wandb.name": None,
    "++logger.wandb.entity": None,
}
SPECDEC_COMMAND_ASSIGNMENTS = {
    "++policy.generation.vllm_kwargs.speculative_config.method": "eagle3",
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens": "5",
    "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size": (  # noqa: E501
        "1"
    ),
    "++policy.generation.vllm_kwargs.speculative_config.rejection_sample_method": (
        "standard"
    ),
    "++policy.generation.vllm_kwargs.speculative_config.draft_sample_method": (
        "probabilistic"
    ),
}
THRESHOLD_COMMAND_ASSIGNMENTS = {
    "++policy.generation.vllm_kwargs.scheduler_cls": (
        "nemo_rl.models.generation.vllm.tail_gate_scheduler.TailGatedScheduler"
    ),
    "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_mode": (
        "threshold"
    ),
    "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_consecutive_checks": (  # noqa: E501
        "10"
    ),
    "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_off_mode": (
        "advance_only"
    ),
}
MINI_LAUNCHER_ENV_ASSIGNMENTS: dict[str, str | None] = {
    "CONTAINER": None,
    "MOUNTS": "/lustre:/lustre",
    "CONTAINER_WORKDIR": None,
    "COMMAND": None,
    "BASE_LOG_DIR": None,
    "GPUS_PER_NODE": "4",
    "HF_HOME": None,
    "PYTHONPATH": None,
    "PYTHONDONTWRITEBYTECODE": "1",
    "RAY_LOG_SYNC_FREQUENCY": "60",
    "TMPDIR": "/tmp",
    "TRITON_CACHE_DIR": None,
    "TORCHINDUCTOR_CACHE_DIR": None,
}
MINI_SBATCH_OPTIONS: dict[str, str | None] = {
    "--parsable": "",
    "--account": None,
    "--partition": None,
    "--nodes": "4",
    "--ntasks-per-node": "1",
    "--exclusive": "",
    "--time": None,
    "--segment": "4",
    "--job-name": None,
    "--output": None,
    "--open-mode": "append",
    "--comment": "metrics",
}
FINAL_SYNC_MARKER = ".ray_logs_final_sync_complete"
FINAL_SYNC_EVIDENCE_DIR = ".ray_logs_final_sync_evidence"
FINAL_SYNC_EVIDENCE_FILES = frozenset(
    {"head", *(f"worker-{worker}" for worker in range(MINI_EXPECTED_NODES - 1))}
)
TEXT_LOG_SUFFIXES = {".err", ".log", ".out", ".txt"}
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOG_FAILURE_PATTERNS = (
    (
        "stale_draft_id",
        re.compile(
            r"(?:^\s*(?:[\w.]*?(?:runtimeerror|valueerror|assertionerror|"
            r"exception)|error)\s*:.*\bstale draft (?:ids?|token ids?)\b|"
            r"\bstale draft (?:ids?|token ids?)\s*"
            r"(?:(?:observed|found|detected)\s*)?[:=]\s*"
            r"(?:[1-9]\d*|\[[^\]]*[1-9]\d*[^\]]*\]))",
            re.IGNORECASE,
        ),
    ),
    (
        "invalid_token",
        re.compile(
            r"(?:^\s*(?:[\w.]*?(?:runtimeerror|valueerror|assertionerror|"
            r"exception)|error)\s*:.*\binvalid tokens?(?: ids?)?\b|"
            r"\binvalid tokens?(?: ids?)?\s*"
            r"(?:(?:observed|found|detected)\s*)?[:=]\s*"
            r"(?:[1-9]\d*|\[[^\]]*[1-9]\d*[^\]]*\]))",
            re.IGNORECASE,
        ),
    ),
    (
        "tokens_left_for_obs",
        re.compile(r"\btokens_left_for_obs\s*[:=]\s*-[1-9]\d*\b", re.IGNORECASE),
    ),
    (
        "nan",
        re.compile(
            r"(?:\b(?:loss|reward|logprobs?|gradients?|metrics?)\b\s*"
            r"(?:is|are|=|:|contains?)\s*nan\b|\bnan detected in\b.*"
            r"\b(?:loss|reward|logprobs?|gradients?|metrics?)\b|"
            r"\b(?:found|detected|encountered)\s+nan\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "oom",
        re.compile(
            r"(?:^\s*slurmstepd(?:\[[^\]]+\])?:\s*error:.*"
            r"(?:\boom[_ -]?kill\b|\boom\b|\bout of memory\b)|"
            r"\b(?:cuda out of memory|outofmemoryerror)\b|"
            r"^\s*(?:runtimeerror|error|exception)\s*:.*\boom\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "nccl",
        re.compile(
            r"(?:\bdistbackenderror\s*:.*\bnccl\b.*"
            r"\b(?:error|timed out|timeout|hang|hung|aborted)\b|"
            r"^\s*runtimeerror\s*:\s*nccl error\s*:.*\b"
            r"(?:unhandled system error|timed out|timeout|hang|hung|aborted)\b|"
            r"\bnccl\b.*\b(?:watchdog\s+timed out|timeout detected|"
            r"hang detected|hung|aborted)\b|\bwatchdog caught collective "
            r"operation timeout\b.*\bworknccl\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "q_cache",
        re.compile(
            r"(?:^\s*(?:[\w.]*?(?:runtimeerror|valueerror|assertionerror|"
            r"exception)|error)\s*:.*\bq[-_ ]?cache\b|\bq[-_ ]?cache\b.*"
            r"\b(?:mismatch|failure|corruption)\s+(?:detected|found|failed)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "cuda_graph_fallback",
        re.compile(
            r"(?:^\s*(?:(?:\[[^\]\n]+\]|\([^\)\n]+\)|ray::[^\s:]+)\s*:?\s*)*"
            r"(?:runtimeerror|indexerror|error)\s*:.*\bcuda[ _]?graphs?\b.*"
            r"\b(?:fallback|capture|replay|execution|failed)\b|"
            r"\bcuda[ _]?graphs?\s+fallback\s+"
            r"(?:to eager|detected|used|occurred)\b|"
            r"\bcuda[ _]?graphs?\s+fallback count\s*[:=]\s*[1-9]\d*\b|"
            r"\b(?:vllm:)?cudagraph(?:_[a-z0-9]+)*_"
            r"(?:eager_)?fallback(?:_count)?\s*[:=]\s*[1-9]\d*(?:\.0+)?\b|"
            r"\beager[ _-]?fallback(?:[ _]?count)?\s*[:=]\s*[1-9]\d*\b)",
            re.IGNORECASE,
        ),
    ),
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _create_wandb_api() -> WandbApi:
    wandb = importlib.import_module("wandb")
    return cast(WandbApi, wandb.Api())


def _wandb_run_path_from_url(url: str, *, variant: str, expected_run_id: str) -> str:
    parsed = urlparse(url)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or len(parts) != 4
        or parts[2] != "runs"
        or not parts[0]
        or not parts[1]
        or not parts[3]
    ):
        raise ValueError(f"invalid wandb_url:{variant}:{url}")
    entity, project, _, run_id = parts
    if run_id != expected_run_id:
        raise ValueError(
            f"wandb_url run ID mismatch:{variant}:{run_id}:{expected_run_id}"
        )
    return f"{entity}/{project}/{run_id}"


def _split_assignment(
    token: str, *, allow_hydra_prefix: bool
) -> tuple[str, str] | None:
    candidate = token
    if allow_hydra_prefix and candidate.startswith("++"):
        candidate = candidate[2:]
    if "=" not in candidate:
        return None
    name, value = candidate.split("=", maxsplit=1)
    if not name or (not allow_hydra_prefix and not re.fullmatch(r"[A-Za-z_]\w*", name)):
        return None
    return name, value


def _assignment_error(
    assignments: Mapping[str, str],
    expected: Mapping[str, str | None],
) -> str | None:
    if unknown := sorted(assignments.keys() - expected.keys()):
        return f"unknown:{unknown[0]}"
    if missing := sorted(expected.keys() - assignments.keys()):
        return f"missing:{missing[0]}"
    for name, expected_value in expected.items():
        actual = assignments[name]
        if (expected_value is None and not actual) or (
            expected_value is not None and actual != expected_value
        ):
            return name
    return None


def _wandb_identity(row: Mapping[str, str]) -> tuple[str, str]:
    url = row.get("wandb_url", "")
    if not url:
        return DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT
    parts = [unquote(part) for part in urlparse(url).path.split("/") if part]
    if len(parts) == 4 and parts[2] == "runs":
        return parts[0], parts[1]
    return "", ""


def _expected_command_assignments(row: Mapping[str, str]) -> dict[str, str | None]:
    wandb_entity, wandb_project = _wandb_identity(row)
    expected: dict[str, str | None] = {
        **MINI_COMMAND_ASSIGNMENTS,
        "checkpointing.checkpoint_dir": f"{row['run_dir']}/checkpoints",
        "logger.wandb.project": wandb_project,
        "logger.wandb.name": row["wandb_run_id"],
        "++logger.wandb.entity": wandb_entity,
        "logger.log_dir": f"{row['run_dir']}/nemo_logs",
    }
    if row["variant"] != "baseline_v2":
        expected.update(SPECDEC_COMMAND_ASSIGNMENTS)
        expected["++policy.generation.vllm_kwargs.speculative_config.model"] = row[
            "draft_checkpoint"
        ]
    if row["variant"] == "fastrl_threshold_v2_k5":
        expected.update(THRESHOLD_COMMAND_ASSIGNMENTS)
        expected[
            "++policy.generation.vllm_kwargs.speculative_config.sd_tail_gate_threshold"
        ] = row["threshold"]
    return expected


def _contains_shell_active_syntax(value: str) -> bool:
    return any(token in value for token in ("$(", "`", ";", "&&", "||", "|", "<", ">"))


def _structured_argv(
    row: Mapping[str, str], *, display_field: str, argv_field: str, label: str
) -> tuple[list[str] | None, str | None]:
    variant = row["variant"]
    try:
        raw_argv = json.loads(row[argv_field])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None, f"invalid mini {label}:{variant}:argv"
    if not isinstance(raw_argv, list) or not all(
        isinstance(token, str) for token in raw_argv
    ):
        return None, f"invalid mini {label}:{variant}:argv"
    tokens = cast(list[str], raw_argv)
    display = row.get(display_field, "")
    if _contains_shell_active_syntax(display) or any(
        _contains_shell_active_syntax(token) for token in tokens
    ):
        return None, f"invalid mini {label}:{variant}:argv"
    try:
        display_tokens = shlex.split(display)
    except ValueError:
        return None, f"invalid mini {label}:{variant}:argv"
    if display_tokens != tokens:
        return None, f"invalid mini {label}:{variant}:argv"
    return tokens, None


def _mini_command_error(row: Mapping[str, str]) -> str | None:
    variant = row["variant"]
    tokens, argv_error = _structured_argv(
        row,
        display_field="command",
        argv_field="command_argv_json",
        label="command",
    )
    if argv_error:
        return argv_error
    assert tokens is not None
    if not tokens or tokens[0] != "env":
        return f"invalid mini command:{variant}:shape"

    environment: dict[str, str] = {}
    index = 1
    while index < len(tokens) and tokens[index] != "uv":
        assignment = _split_assignment(tokens[index], allow_hydra_prefix=False)
        if assignment is None:
            return f"invalid mini command:{variant}:entrypoint"
        name, value = assignment
        if name in environment:
            return f"invalid mini command:{variant}:duplicate:{name}"
        environment[name] = value
        index += 1
    expected_environment = {
        **MINI_COMMAND_ENV_ASSIGNMENTS,
        "WANDB_RUN_ID": row["wandb_run_id"],
        "PYTHONPATH": row["checkout_path"],
    }
    if error := _assignment_error(environment, expected_environment):
        return f"invalid mini command:{variant}:environment:{error}"

    executable = ["uv", "run", "examples/run_grpo.py"]
    if tokens[index : index + len(executable)] != executable:
        return f"invalid mini command:{variant}:entrypoint"
    index += len(executable)
    if tokens[index : index + 2] != ["--config", row["recipe"]]:
        return f"invalid mini command:{variant}:recipe"
    index += 2

    assignments: dict[str, str] = {}
    normalized_names: set[str] = set()
    for token in tokens[index:]:
        assignment = _split_assignment(token, allow_hydra_prefix=True)
        if assignment is None:
            return f"invalid mini command:{variant}:override:{token}"
        normalized_name, value = assignment
        if normalized_name in normalized_names:
            return f"invalid mini command:{variant}:duplicate:{normalized_name}"
        normalized_names.add(normalized_name)
        key = f"++{normalized_name}" if token.startswith("++") else normalized_name
        assignments[key] = value
    if error := _assignment_error(assignments, _expected_command_assignments(row)):
        if error == "++policy.generation.vllm_kwargs.speculative_config.model":
            return f"invalid mini command:{variant}:provenance:draft_checkpoint"
        return f"invalid mini command:{variant}:override:{error}"
    return None


def _mini_launcher_command_error(row: Mapping[str, str]) -> str | None:
    variant = row["variant"]
    tokens, argv_error = _structured_argv(
        row,
        display_field="launcher_command",
        argv_field="launcher_argv_json",
        label="launcher command",
    )
    if argv_error:
        return argv_error
    assert tokens is not None
    if not tokens or tokens[0] != "env":
        return f"invalid mini launcher command:{variant}:shape"

    environment: dict[str, str] = {}
    index = 1
    while index < len(tokens) and tokens[index] != "sbatch":
        assignment = _split_assignment(tokens[index], allow_hydra_prefix=False)
        if assignment is None:
            return f"invalid mini launcher command:{variant}:shape"
        name, value = assignment
        if name in environment:
            return f"invalid mini launcher command:{variant}:duplicate:{name}"
        environment[name] = value
        index += 1
    if index >= len(tokens) or tokens[index] != "sbatch":
        return f"invalid mini launcher command:{variant}:shape"
    expected_environment = {
        **MINI_LAUNCHER_ENV_ASSIGNMENTS,
        "CONTAINER": row["container"],
        "CONTAINER_WORKDIR": row["checkout_path"],
        "BASE_LOG_DIR": row["run_dir"],
        "COMMAND": row["command"],
        "PYTHONPATH": row["checkout_path"],
    }
    if error := _assignment_error(environment, expected_environment):
        provenance_field = {
            "CONTAINER": "container",
            "CONTAINER_WORKDIR": "checkout_path",
            "PYTHONPATH": "checkout_path",
        }.get(error)
        if provenance_field:
            return (
                f"invalid mini launcher command:{variant}:provenance:{provenance_field}"
            )
        return f"invalid mini launcher command:{variant}:environment:{error}"

    index += 1
    if index >= len(tokens) or tokens[-1] != row["ray_sub_path"]:
        return f"invalid mini launcher command:{variant}:provenance:ray_sub_path"
    option_tokens = tokens[index:-1]
    options: dict[str, str] = {}
    for token in option_tokens:
        if not token.startswith("--"):
            return f"invalid mini launcher command:{variant}:sbatch options"
        if "=" in token:
            name, value = token.split("=", maxsplit=1)
        else:
            name, value = token, ""
        if name in options:
            return f"invalid mini launcher command:{variant}:duplicate:{name}"
        options[name] = value
    expected_options = {
        **MINI_SBATCH_OPTIONS,
        "--output": f"{row['run_dir']}/slurm-%j.out",
    }
    if error := _assignment_error(options, expected_options):
        return f"invalid mini launcher command:{variant}:sbatch options:{error}"
    return None


def _mini_execution_provenance_error(row: Mapping[str, str]) -> str | None:
    variant = row["variant"]
    checkout_path = Path(row["checkout_path"])
    ray_sub_path = Path(row["ray_sub_path"])
    container_path = Path(row["container"])
    if not checkout_path.is_absolute() or not container_path.is_absolute():
        return f"invalid mini provenance:{variant}:absolute_paths"
    if ray_sub_path != checkout_path / "ray.sub":
        return f"invalid mini provenance:{variant}:ray_sub_path"
    if not ray_sub_path.is_absolute():
        return f"invalid mini provenance:{variant}:ray_sub_path"
    if row["variant"] == "baseline_v2":
        if row["draft_checkpoint"] != "not_applicable":
            return f"invalid mini provenance:{variant}:draft_checkpoint"
    elif not Path(row["draft_checkpoint"]).is_absolute():
        return f"invalid mini provenance:{variant}:draft_checkpoint"
    return None


def _resolved_manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve(strict=False)


def _mini_log_provenance_error(manifest: Path, row: Mapping[str, str]) -> str | None:
    variant = row["variant"]
    run_dir = _resolved_manifest_path(manifest, row["run_dir"])
    expected_run_dir = (manifest.parent / row["model"] / variant).resolve(strict=False)
    if run_dir != expected_run_dir:
        return f"invalid mini log provenance:{variant}:run_dir"
    job_log_dir = run_dir / f"{row['job_id']}-logs"
    expected_paths = {
        "slurm_log_path": run_dir / f"slurm-{row['job_id']}.out",
        "ray_driver_log_path": job_log_dir / "ray-driver.log",
        "ray_log_dir": job_log_dir / "ray",
    }
    for field, expected in expected_paths.items():
        if _resolved_manifest_path(manifest, row[field]) != expected:
            return f"invalid mini log provenance:{variant}:{field}"
    return None


def _mini_capacity_error(rows: Iterable[Mapping[str, str]]) -> str | None:
    capacities: dict[str, int] = {}
    for row in rows:
        variant = row["variant"]
        try:
            num_prompts = int(row["num_prompts"])
            num_generations = int(row["num_generations"])
            dp = int(row["dp"])
        except ValueError:
            return f"invalid mini scheduler capacity:{variant}:non-integer"
        if num_prompts <= 0 or num_generations <= 0 or dp <= 0:
            return f"invalid mini scheduler capacity:{variant}:non-positive"
        global_rollouts = num_prompts * num_generations
        if global_rollouts % dp:
            return (
                f"invalid mini scheduler capacity:{variant}:"
                f"global rollouts not divisible by dp:{global_rollouts}:{dp}"
            )
        capacity = global_rollouts // dp
        if capacity != MINI_EXPECTED_LOCAL_CAPACITY:
            return (
                f"invalid mini local scheduler capacity:{variant}:{capacity}:"
                f"expected:{MINI_EXPECTED_LOCAL_CAPACITY}"
            )
        if global_rollouts != MINI_EXPECTED_GLOBAL_ROLLOUTS:
            return (
                f"invalid mini global rollouts:{variant}:{global_rollouts}:"
                f"expected:{MINI_EXPECTED_GLOBAL_ROLLOUTS}"
            )
        capacities[variant] = capacity
    threshold_row = next(
        row for row in rows if row["variant"] == "fastrl_threshold_v2_k5"
    )
    try:
        threshold = int(threshold_row["threshold"])
    except ValueError:
        return "invalid mini threshold:non-integer"
    capacity = capacities["fastrl_threshold_v2_k5"]
    if not 0 < threshold < capacity:
        return f"invalid mini threshold:{threshold}:capacity:{capacity}"
    if threshold != MINI_EXPECTED_THRESHOLD:
        return f"invalid mini threshold:{threshold}:expected:{MINI_EXPECTED_THRESHOLD}"
    return None


def _validate_mini_manifest_rows(
    rows: list[dict[str, str]], *, manifest: Path
) -> str | None:
    for row in rows:
        missing = [
            field for field in MINI_REQUIRED_MANIFEST_FIELDS if not row.get(field)
        ]
        if missing:
            return f"missing mini manifest fields:{','.join(missing)}"
    variants = sorted(row.get("variant", "") for row in rows)
    required_variants = sorted(REQUIRED_VARIANT_CONFIG)
    if variants != required_variants:
        return (
            "mini manifest variants must be exactly:"
            f"{','.join(required_variants)}:got:{','.join(variants)}"
        )
    if capacity_error := _mini_capacity_error(rows):
        return capacity_error
    for row in rows:
        variant = row["variant"]
        for field, expected in REQUIRED_COMMON_CONFIG.items():
            actual = row.get(field, "")
            if actual != expected:
                return (
                    f"invalid mini manifest field:{variant}:{field}:{actual}:{expected}"
                )
        for field, expected in REQUIRED_VARIANT_CONFIG[variant].items():
            actual = row.get(field, "")
            if actual != expected:
                return (
                    f"invalid mini manifest field:{variant}:{field}:{actual}:{expected}"
                )
        checkpointing_enabled = row.get("checkpointing_enabled", "")
        if checkpointing_enabled and checkpointing_enabled.lower() != "false":
            return f"invalid mini manifest provenance:{variant}:checkpointing_enabled"
        if row.get("wandb_url"):
            try:
                _wandb_run_path_from_url(
                    row["wandb_url"],
                    variant=variant,
                    expected_run_id=row["wandb_run_id"],
                )
            except ValueError as error:
                return str(error)
        if command_error := _mini_command_error(row):
            return command_error
        if launcher_error := _mini_launcher_command_error(row):
            return launcher_error
        if provenance_error := _mini_execution_provenance_error(row):
            return provenance_error
        if log_error := _mini_log_provenance_error(manifest, row):
            return log_error
    return None


def _mini_threshold(rows: Iterable[Mapping[str, str]]) -> int:
    threshold_row = next(
        row for row in rows if row["variant"] == "fastrl_threshold_v2_k5"
    )
    return int(threshold_row["threshold"])


def _wandb_run_path(
    metadata: Mapping[str, str], *, fallback_entity: str, fallback_project: str
) -> str:
    url = metadata.get("wandb_url", "")
    if url:
        return _wandb_run_path_from_url(
            url,
            variant=metadata["variant"],
            expected_run_id=metadata["wandb_run_id"],
        )
    return f"{fallback_entity}/{fallback_project}/{metadata['wandb_run_id']}"


def _records_by_step(
    history: Iterable[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    return {
        step: record
        for record in history
        if isinstance((step := record.get("_step")), int)
        and not isinstance(step, bool)
        and step in MINI_STEPS
    }


def _positive_metric(record: Mapping[str, object], key: str) -> bool:
    value = record.get(key)
    return _is_finite_number(value) and value > 0.0


def _scan_log(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                normalized_line = ANSI_ESCAPE_PATTERN.sub("", line)
                for reason, pattern in LOG_FAILURE_PATTERNS:
                    if pattern.search(normalized_line):
                        return f"logs:{reason}"
    except OSError:
        return "log_unreadable"
    return None


def _ray_attempt_dirs(run_dir: Path, job_id: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(job_id)}(?:-(\d+))?-logs$")
    attempts: list[tuple[int, str, Path]] = []
    try:
        children = list(run_dir.iterdir())
    except OSError:
        return []
    for path in children:
        match = pattern.fullmatch(path.name)
        if match and path.is_dir():
            restart = int(match.group(1)) if match.group(1) is not None else -1
            attempts.append((restart, path.name, path))
    return [path for _, _, path in sorted(attempts)]


def _log_health_failure(manifest: Path, metadata: Mapping[str, str]) -> str | None:
    slurm_log = _resolved_manifest_path(manifest, metadata["slurm_log_path"])
    if not slurm_log.is_file():
        return "log_missing:slurm_log_path"

    run_dir = _resolved_manifest_path(manifest, metadata["run_dir"])
    attempts = _ray_attempt_dirs(run_dir, metadata["job_id"])
    if not attempts:
        return "log_missing:ray_log_dir"
    final_attempt = attempts[-1]
    final_driver = final_attempt / "ray-driver.log"
    if not final_driver.is_file():
        return "log_missing:ray_driver_log_path"
    final_ray_dir = final_attempt / "ray"
    if not final_ray_dir.is_dir():
        return "log_missing:ray_log_dir"
    final_ray_logs = sorted(
        path
        for path in final_ray_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_LOG_SUFFIXES
    )
    if not final_ray_logs:
        return "log_empty:ray_log_dir"
    if not (final_attempt / FINAL_SYNC_MARKER).is_file():
        return "log_missing:final_sync_marker"
    evidence_dir = final_attempt / FINAL_SYNC_EVIDENCE_DIR
    if not evidence_dir.is_dir():
        return "log_missing:final_sync_node_evidence"
    evidence_entries = list(evidence_dir.iterdir())
    if {
        entry.name for entry in evidence_entries
    } != FINAL_SYNC_EVIDENCE_FILES or not all(
        entry.is_file() for entry in evidence_entries
    ):
        return "log_missing:final_sync_node_evidence"

    text_logs = [slurm_log]
    for attempt in attempts:
        text_logs.extend(
            sorted(
                path
                for path in attempt.rglob("*")
                if path.is_file() and path.suffix.lower() in TEXT_LOG_SUFFIXES
            )
        )
    for path in text_logs:
        if failure := _scan_log(path):
            return failure
    return None


def _mini_failure(
    summary: RunSummary,
    history: Iterable[Mapping[str, object]],
    metadata: Mapping[str, str],
    *,
    threshold: int,
    log_failure: str | None,
) -> str | None:
    if summary.status != "final":
        return summary.reason
    if log_failure is not None:
        return log_failure
    records = _records_by_step(history)
    if set(records) != MINI_STEPS:
        return (
            f"missing_steps:{','.join(map(str, sorted(MINI_STEPS - records.keys())))}"
        )
    for record in records.values():
        if not _positive_metric(record, "timing/train/policy_training"):
            return "policy_training"
        if not _positive_metric(record, "timing/train/policy_and_reference_logprobs"):
            return "policy_and_reference_logprobs"

    if metadata.get("gate_mode") != "threshold":
        return None
    for record in records.values():
        checks = (
            (
                "tail_gate_activations",
                "train/vllm/tail_gate_activations",
                lambda value: value == 1.0,
            ),
            (
                "activation_tick",
                "train/vllm/tail_gate_activation_tick",
                lambda value: value > 0.0,
            ),
            (
                "activation_batch",
                "train/vllm/tail_gate_activation_batch",
                lambda value: 1.0 <= value <= threshold,
            ),
            (
                "gate_enabled_ratio",
                "train/vllm/tail_gate_enabled_step_ratio",
                lambda value: 0.0 < value < 1.0,
            ),
            (
                "gate_advance_only_ratio",
                "train/vllm/tail_gate_advance_only_step_ratio",
                lambda value: 0.0 < value < 1.0,
            ),
            (
                "tail_gate_k0_steps",
                MINI_METRIC_KEYS["tail_gate_k0_steps"],
                lambda value: value > 0.0,
            ),
            (
                "tail_gate_k5_steps",
                MINI_METRIC_KEYS["tail_gate_k5_steps"],
                lambda value: value > 0.0,
            ),
            ("num_drafts", "train/vllm/spec_num_drafts", lambda value: value > 0.0),
            (
                "num_accepted_tokens",
                "train/vllm/spec_num_accepted_tokens",
                lambda value: value > 0.0,
            ),
        )
        for name, key, predicate in checks:
            value = record.get(key)
            if not _is_finite_number(value) or not predicate(float(value)):
                return name
    return None


def _activation_events(
    metadata: Mapping[str, str], history: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    if metadata.get("gate_mode") != "threshold":
        return []
    events: list[dict[str, object]] = []
    for step, record in sorted(_records_by_step(history).items()):
        tick = record.get("train/vllm/tail_gate_activation_tick")
        batch = record.get("train/vllm/tail_gate_activation_batch")
        activations = record.get("train/vllm/tail_gate_activations")
        if (
            _is_finite_number(tick)
            and _is_finite_number(batch)
            and _is_finite_number(activations)
            and activations > 0.0
        ):
            events.append(
                {
                    "job_id": metadata["job_id"],
                    "step": step,
                    "tick": float(tick),
                    "batch": float(batch),
                    "variant": metadata["variant"],
                }
            )
    return events


def _render_activation_scatter(
    events: list[dict[str, object]], *, threshold: int
) -> str:
    ordered = sorted(
        events,
        key=lambda event: (
            cast(str, event["variant"]),
            cast(str, event["job_id"]),
            cast(int, event["step"]),
            cast(float, event["tick"]),
            cast(float, event["batch"]),
        ),
    )
    width = 460
    height = 220
    left = 52
    top = 18
    plot_width = 382
    plot_height = 152
    max_tick = max(
        [float(threshold), *(cast(float, event["tick"]) for event in ordered)]
    )
    max_batch = max(
        [float(threshold), *(cast(float, event["batch"]) for event in ordered)]
    )

    def x(value: float) -> float:
        return left + plot_width * value / max_tick

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value / max_batch)

    threshold_y = y(float(threshold))
    fragments = [
        '<section class="tail-gate-activation-events">',
        "<style>.tail-gate-activation-events{font:13px sans-serif}.tail-gate-activation-events svg{border:1px solid #c9c9c9}.tail-gate-activation-events .axis{stroke:#333}.tail-gate-activation-events .threshold{stroke:#c55;stroke-dasharray:4 3}.tail-gate-activation-events .event{fill:#76b900}.tail-gate-activation-events text{fill:#222}</style>",
        "<p>This two-step smoke makes no speedup claim.</p>",
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Tail-gate activation events">',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<line class="threshold" x1="{left}" y1="{threshold_y:.1f}" x2="{left + plot_width}" y2="{threshold_y:.1f}"/>',
        f'<text x="{left + 4}" y="{threshold_y - 4:.1f}">threshold={threshold}</text>',
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle">Scheduler tick</text>',
        f'<text x="14" y="{top + plot_height / 2:.1f}" transform="rotate(-90 14 {top + plot_height / 2:.1f})" text-anchor="middle">Inflight batch</text>',
    ]
    for event in ordered:
        tick = cast(float, event["tick"])
        batch = cast(float, event["batch"])
        label = (
            f"OFF-to-ON: {html.escape(cast(str, event['variant']))} "
            f"step={cast(int, event['step'])} tick={tick:g} batch={batch:g}"
        )
        fragments.append(
            f'<circle class="event" cx="{x(tick):.1f}" cy="{y(batch):.1f}" r="4"><title>{label}</title></circle>'
        )
        fragments.append(
            f'<text x="{x(tick) + 6:.1f}" y="{y(batch) - 6:.1f}">{label}</text>'
        )
    fragments.extend(["</svg>", "</section>\n"])
    return "\n".join(fragments)


def _mini_row(
    comparison: ComparisonRow,
    failure: str | None,
    history: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    row = comparison.to_dict()
    records = _records_by_step(history)
    row["mini_health_passed"] = row["status"] == "final" and failure is None
    for name, key in MINI_METRIC_KEYS.items():
        values = [record.get(key) for record in records.values()]
        finite_values = [float(value) for value in values if _is_finite_number(value)]
        row[name] = (
            statistics.fmean(finite_values)
            if len(finite_values) == len(MINI_STEPS)
            else None
        )
    return row


def main(argv: list[str] | None = None, *, api: WandbApi | None = None) -> int:
    """Validate all mini-smoke manifest rows and render deterministic artifacts."""
    args = _parse_args(argv)
    _, manifest_rows = _read_manifest(args.manifest)
    manifest_error = _validate_manifest_rows(manifest_rows)
    if manifest_error:
        raise ValueError(manifest_error)
    mini_manifest_error = _validate_mini_manifest_rows(
        manifest_rows, manifest=args.manifest
    )
    if mini_manifest_error:
        raise ValueError(mini_manifest_error)
    threshold = _mini_threshold(manifest_rows)
    _claim_output_directory(args.output_dir)

    client = api if api is not None else _create_wandb_api()
    summaries: list[RunSummary] = []
    histories: dict[str, list[Mapping[str, object]]] = {}
    failures: dict[str, str | None] = {}
    events: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        metadata = {
            **manifest_row,
            "source": manifest_row.get("source") or args.manifest.name,
        }
        job_id = metadata["job_id"]
        try:
            run = client.run(
                _wandb_run_path(
                    metadata,
                    fallback_entity=args.entity,
                    fallback_project=args.project,
                )
            )
            if not metadata.get("wandb_url"):
                metadata["wandb_url"] = run.url
            history = list(
                run.scan_history(
                    keys=[*_history_keys(metadata), *MINI_METRIC_KEYS.values()]
                )
            )
            histories[job_id] = history
            summary = summarize_history(metadata, history, expected_steps=MINI_STEPS)
            failure = _mini_failure(
                summary,
                history,
                metadata,
                threshold=threshold,
                log_failure=_log_health_failure(args.manifest, metadata),
            )
            if summary.status == "final" and failure is not None:
                summary = replace(
                    summary,
                    status="health_failed",
                    reason=f"mini_health_failed:{failure}",
                )
            summaries.append(summary)
            failures[job_id] = failure
            events.extend(_activation_events(metadata, history))
        except Exception as error:  # Preserve every failed W&B row in the report.
            summaries.append(
                _empty_summary(
                    metadata, f"wandb_fetch_failed:{type(error).__name__}", []
                )
            )
            histories[job_id] = []
            failures[job_id] = f"wandb_fetch_failed:{type(error).__name__}"

    try:
        comparisons = build_comparison_rows(summaries)
    except ValueError as error:
        comparisons = [
            _empty_comparison_row(
                replace(summary, status="partial", reason=f"comparison_failed:{error}")
            )
            for summary in summaries
        ]
    rows = [
        _mini_row(
            comparison,
            failures[comparison.summary.job_id],
            histories[comparison.summary.job_id],
        )
        for comparison in sorted(
            comparisons,
            key=lambda comparison: (
                comparison.summary.runner,
                comparison.summary.model,
                comparison.summary.variant,
                comparison.summary.job_id,
            ),
        )
    ]
    _write_atomic(
        args.output_dir / "mini_summary.json",
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(args.output_dir / "mini_summary.csv", rows, fieldnames=MINI_ROW_FIELDS)
    _write_atomic(
        args.output_dir / "tail_gate_activation_events.html",
        _render_activation_scatter(events, threshold=threshold),
    )
    return int(
        any(row["status"] != "final" or not row["mini_health_passed"] for row in rows)
    )


if __name__ == "__main__":
    raise SystemExit(main())
