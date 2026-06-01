#!/usr/bin/env python3
"""Validate the SpecDec-RL remote patch overlay for Qwen3 Eagle3 rollout.

This is a no-submit, mostly local check. It proves that the handoff contains the
runtime compatibility files needed for the current Qwen3-235B SWE rollout path
without claiming that a remote SpecDec-RL checkout has already been patched.
When a target checkout is visible, it also reports whether the overlay appears
to be applied there.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_PATCH_ROOT = EXP / "remote_patches" / "SpecDec-RL"
DEFAULT_TARGET = Path("/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL")

REQUIRED_FILES: dict[str, list[str]] = {
    "ray.sub": [
        "maybe_gres_arg()",
        'partition_gres="$(sinfo -p "$SLURM_JOB_PARTITION"',
        'COMMON_SRUN_ARGS="$GRES_ARG"',
        "CPUS_PER_WORKER=${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}",
        "RAY_VERSION=${RAY_VERSION:-2.49.2}",
    ],
    "nemo_rl/models/generation/vllm/vllm_worker_async.py": [
        "model_config = self.llm_async_engine_args.create_model_config()",
        'openai_serving_models_kwargs["model_config"] = model_config',
        "spec_decode_enabled = bool(llm_kwargs.get(\"speculative_config\"))",
        "or spec_decode_enabled",
        "request.logprobs = False",
        "self._vllm_spec_decode_metrics_baseline",
        "self._vllm_specdec_gate_metrics_baseline",
        "spec_decode_gate",
    ],
    "nemo_rl/models/generation/vllm/vllm_worker.py": [
        "def _patch_vllm_speculative_decoding_post_step(required: bool)",
        "def _patch_vllm_batch_gated_speculative_decoding()",
        "def _patch_vllm_adaptive_specdec_gate()",
        "NRL_SPECDEC_BATCH_GATE_PATCH_V7",
        "NRL_SPECDEC_ADAPTIVE_GATE_PATCH_V1",
        "NRL_SPECDEC_SCHEDULER_LOOKAHEAD_GATE_PATCH_V5",
        "NRL_SPECDEC_SCHEDULER_ADAPTIVE_GATE_PATCH_V1",
        "NRL_SPECDEC_SCHEDULER_DYNAMIC_DRAFT_CAP_PATCH_V1",
        "_nrl_specdec_scheduler_lookahead_tokens",
        "partial vLLM adaptive SpecDec runner-gate",
        "partial vLLM adaptive SpecDec scheduler-gate",
        "adaptive scheduler lookahead call",
        "VLLM_SPECDEC_ADAPTIVE_GATE_MODE",
        "VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO",
        "VLLM_SPECDEC_DYNAMIC_DRAFT_TOKENS",
        "VLLM_SPECDEC_BATCH_SIZE_GATE_THRESHOLD",
        "VLLM_SPECDEC_BATCH_TOKEN_GATE_THRESHOLD",
        "_nrl_specdec_scheduler_dynamic_last_selected_tokens",
        "_nrl_specdec_scheduler_dynamic_last_selected_tier",
        "_nrl_specdec_scheduler_dynamic_selected_by_request",
        "_nrl_specdec_scheduler_dynamic_small_selected_count",
        "_nrl_specdec_scheduler_dynamic_small_selected_token_count",
        "_nrl_specdec_scheduler_dynamic_pos1_selected_count",
        "_nrl_specdec_scheduler_dynamic_pos8_selected_count",
        "NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_COUNTERS_ON_STORE_V1",
        "NRL_SPECDEC_SCHEDULER_DYNAMIC_SELECTED_BY_REQUEST_V1",
        "NRL_SPECDEC_SCHEDULER_STORE_COUNTERS_BEFORE_PER_REQUEST_V1",
        "NRL_SPECDEC_SCHEDULER_DYNAMIC_POS_COUNTERS_PARTIAL_UPGRADE_V1",
        "NRL_SPECDEC_SCHEDULER_REQUEST_ID_ARITY_GUARD_V1",
        "NRL_SPECDEC_DYNAMIC_STORE_COUNTERS_DIFF_V1",
        "_nrl_specdec_batch_gate_threshold",
        "_nrl_specdec_batch_gate_token_threshold",
        "specdec_batch_gate_num_requests",
        "specdec_batch_gate_num_tokens",
        "specdec_scheduled_token_count",
        "specdec_batch_gate_scheduler_all_disabled",
        "nrl_specdec_batch_gate_all_disabled",
        "nrl_specdec_batch_gate_eligible_count",
        "specdec_scheduled_tokens",
        "scheduled_spec_decode_tokens",
        "VLLM_SPECDEC_ADAPTIVE_MIN_REQUEST_THRESHOLD must be <=",
        "VLLM_SPECDEC_ADAPTIVE_MIN_TOKEN_THRESHOLD must be <=",
        "def _nrl_env_nonnegative_int",
        "def _nrl_env_float",
        "silently become global SpecDec",
        "_read_vllm_specdec_gate_metrics",
        "spec_decode_gate",
        "partial or outdated vLLM SpecDec batch-gate",
        "NRL_ALLOW_SPECDEC_DISABLE_BY_BATCH_SIZE",
        "NRL_ALLOW_SPECDEC_REQUEST_LOGPROBS",
        "VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH",
        "NRL_VLLM_OMIT_GENERATION_LOGPROBS",
        "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS",
        'default_disable_log_stats = "true"',
        "max_new_tokens=allowed_new_tokens_per_sample[idx]",
        "stop_strings=self._merge_stop_strings(",
    ],
    "nemo_rl/models/generation/vllm/vllm_generation.py": [
        "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS",
        "NRL_ALLOW_SPECDEC_REQUEST_LOGPROBS",
        "NRL_VLLM_OMIT_GENERATION_LOGPROBS",
        "VLLM_ENABLE_RUNTIME_SPECDEC_BATCH_GATE_PATCH",
        "VLLM_SPECDEC_ADAPTIVE_GATE_MODE",
        "VLLM_SPECDEC_ADAPTIVE_TARGET_ENABLED_RATIO",
        "spec_decode_gate_totals",
        "spec_decode_totals[\"metrics_available\"] = True",
        "spec_decode_totals[\"metrics_complete\"]",
        "NRL_SPECDEC_ACCEPTANCE_REQUIRES_COMPLETE_DP_METRICS_V1",
        "acceptance_rate_reliable",
        "NRL_SPECDEC_DYNAMIC_STORE_COUNTERS_DP_MERGE_V1",
        "dynamic_small_selected_token_count",
        "f\"dynamic_pos{pos_idx}_selected_count\"",
        "range(1, 9)",
    ],
    "nemo_rl/algorithms/grpo.py": [
        "def _uses_vllm_specdec_without_generation_logprobs(",
        "def _repair_specdec_generation_logprobs_if_safe(",
        "NRL_SPECDEC_CONTROLLER_REQUIRES_COMPLETE_DP_METRICS_V1",
        "controller/action_failed_partial_metrics",
        "next_small = min(max_k, max(next_medium, max(min_k, current_small - 1)))",
        "ray.get(trajectory_collector.set_weight_version.remote(weight_version))",
        "collection_task = trajectory_collector.start_collection.remote(dataloader)",
    ],
    "nemo_rl/algorithms/utils.py": [
        "spec_decode_gate_metrics",
        "spec_decode_gate/{group}/{key}",
        "SpecDec Gate Metrics",
        "spec_decode/acceptance_rate_reliable",
    ],
    "nemo_rl/distributed/virtual_cluster.py": [
        "NRL_RAY_NODE_IP_ADDRESS",
        "RAY_NODE_IP_ADDRESS",
        'ray_init_kwargs["_node_ip_address"] = ray_node_ip_address',
    ],
    "nemo_rl/distributed/ray_actor_environment_registry.py": [
        'getattr(PY_EXECUTABLES, "SGLANG", PY_EXECUTABLES.SYSTEM)',
        'getattr(PY_EXECUTABLES, "FSDP", VLLM_EXECUTABLE)',
        "SGLANG_EXECUTABLE",
        "FSDP_EXECUTABLE",
    ],
    "nemo_rl/models/megatron/community_import.py": [
        "def _patch_distcp_writer_for_ray_import()",
        "def _provider_forwards_expert_tensor_parallel_size(",
        "NRL_MEGATRON_IMPORT_INLINE_WRITER",
        "write_preloaded_data_inline",
        "def _save_megatron_model_with_checkpointing(",
        "def _save_megatron_model(",
        "from megatron.bridge.training.model_load_save import save_megatron_model as save",
        "from megatron.bridge.training.checkpointing import (",
        "MockGPTDatasetConfig",
        "reset_position_ids=False",
        "reset_attention_mask=False",
        "eod_mask_loss=False",
        "model_provider.initialize_model_parallel(seed=0, **model_parallel_kwargs)",
        "provide_distributed_model = getattr(",
        "model_provider.provide_models(wrap_with_ddp=False)",
        "_save_megatron_model(bridge, megatron_model, output_path, hf_model_name, model_provider)",
    ],
    "nemo_rl/models/megatron/setup.py": [
        "from inspect import signature",
        "def calculate_padded_vocab_size(",
        "ProcessGroupCollection = None",
        "def _call_accepts_kwarg(",
        "def _add_pg_collection_kwarg(",
        "_use_mpu_process_groups()",
    ],
    "nemo_rl/models/policy/workers/megatron_policy_worker.py": [
        "try:",
        "from megatron.bridge.training.utils.pg_utils import get_pg_collection",
        "def get_pg_collection(model):",
        "return SimpleNamespace(mp=parallel_state.get_model_parallel_group())",
        "custom_FSDP = ()",
    ],
    "nemo_rl/models/policy/workers/patches.py": [
        "def apply_torch_aten_alias_tensor_patch():",
        "from torch.distributed.tensor._ops._tensor_ops",
        "propagate_single_input_strategy",
    ],
}

IGNORED_DIRS = {"__pycache__", ".git"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument(
        "--target-specdec-rl-dir",
        type=Path,
        default=Path(os.environ.get("SPECDEC_RL_DIR", DEFAULT_TARGET)),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--require-target-applied",
        action="store_true",
        help="Return FAIL unless the visible target checkout already contains the required snippets.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def is_ignored(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    if any(part in IGNORED_DIRS for part in rel.parts):
        return True
    return path.suffix in IGNORED_SUFFIXES


def iter_overlay_files(root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    ignored: list[Path] = []
    if not root.exists():
        return files, ignored
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_ignored(path, root):
            ignored.append(path)
        else:
            files.append(path)
    return files, ignored


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def required_file_rows(patch_root: Path, target_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_visible = target_root.exists()
    for rel, snippets in REQUIRED_FILES.items():
        source = patch_root / rel
        row: dict[str, Any] = {
            "path": rel,
            "source_exists": source.exists(),
            "required_snippet_count": len(snippets),
        }
        if source.exists():
            source_text = read_text(source)
            missing = [snippet for snippet in snippets if snippet not in source_text]
            row.update(
                {
                    "source_sha256": sha256_file(source),
                    "source_size_bytes": source.stat().st_size,
                    "source_missing_snippets": missing,
                    "source_status": "pass" if not missing else "fail",
                }
            )
        else:
            row.update({"source_missing_snippets": snippets, "source_status": "fail"})

        target = target_root / rel
        if not target_visible:
            row["target_status"] = "not_visible"
        elif not target.exists():
            row["target_status"] = "missing"
        else:
            target_text = read_text(target)
            target_missing = [snippet for snippet in snippets if snippet not in target_text]
            target_sha = sha256_file(target)
            row.update(
                {
                    "target_sha256": target_sha,
                    "target_size_bytes": target.stat().st_size,
                    "target_missing_snippets": target_missing,
                    "target_matches_overlay": bool(row.get("source_sha256") and row.get("source_sha256") == target_sha),
                    "target_status": "applied" if not target_missing else "needs_overlay",
                }
            )
        rows.append(row)
    return rows


def check_grpo_generation_indent(checks: list[dict[str, Any]], patch_root: Path) -> None:
    grpo = patch_root / "nemo_rl" / "algorithms" / "grpo.py"
    if not grpo.exists():
        add(checks, "source", "grpo generation timer indent", "fail", f"missing: {grpo}")
        return
    text = read_text(grpo)
    expected = (
        '                if policy_generation is not None and hasattr(\n'
        '                    policy_generation, "clear_vllm_logger_metrics"\n'
        '                ):\n'
        '                    policy_generation.clear_vllm_logger_metrics()\n'
        '\n'
        '                generation_start_s = time.perf_counter()\n'
        '                with timer.time("generation"):\n'
    )
    nested = (
        '                    policy_generation.clear_vllm_logger_metrics()\n'
        '\n'
        '                    with timer.time("generation"):\n'
    )
    if expected in text and nested not in text:
        add(
            checks,
            "source",
            "grpo generation timer indent",
            "pass",
            "generation rollout is not nested under the optional vLLM metrics-clear hook",
        )
    else:
        add(
            checks,
            "source",
            "grpo generation timer indent",
            "fail",
            "generation rollout may be nested under clear_vllm_logger_metrics and skipped for other backends",
        )


def check_python_sources_compile(checks: list[dict[str, Any]], patch_root: Path) -> None:
    py_files = [
        path
        for path in sorted(patch_root.rglob("*.py"))
        if path.is_file() and not is_ignored(path, patch_root)
    ]
    failures: list[str] = []
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(patch_root)}: {exc.msg}")
    if failures:
        add(
            checks,
            "source",
            "overlay python compile",
            "fail",
            "one or more Python overlay files fail py_compile",
            failures=failures[:8],
            failure_count=len(failures),
        )
    else:
        add(
            checks,
            "source",
            "overlay python compile",
            "pass",
            f"py_compile passed for {len(py_files)} Python overlay files",
        )


def check_dynamic_request_id_arity_guard(checks: list[dict[str, Any]], patch_root: Path) -> None:
    worker = patch_root / "nemo_rl" / "models" / "generation" / "vllm" / "vllm_worker.py"
    if not worker.exists():
        add(checks, "source", "dynamic request-id arity guard", "fail", f"missing: {worker}")
        return
    text = read_text(worker)
    required = [
        "NRL_SPECDEC_SCHEDULER_REQUEST_ID_ARITY_GUARD_V1",
        "def _assert_scheduler_dynamic_request_id_arity",
        "expected_signature = (",
        "bad_requestless_call = re.search(",
        "good_request_call = re.search(",
        "request\\.request_id",
        "len\\(num_scheduled_tokens\\)",
    ]
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        add(
            checks,
            "source",
            "dynamic request-id arity guard",
            "fail",
            "vLLM worker does not enforce generated scheduler request_id helper arity",
            missing_snippets=missing,
        )
    else:
        add(
            checks,
            "source",
            "dynamic request-id arity guard",
            "pass",
            "vLLM worker validates generated scheduler helper signature and request_id call sites",
        )


def _eval_string_expr(node: ast.AST, env: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_string_expr(node.left, env) + _eval_string_expr(node.right, env)
    raise ValueError(f"unsupported string expression: {ast.dump(node, include_attributes=False)}")


def _extract_string_assignments(text: str, names: set[str]) -> dict[str, str]:
    tree = ast.parse(text)
    env: dict[str, str] = {}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and hasattr(node, "lineno")
    ]
    assignments.sort(key=lambda node: (node.lineno, node.col_offset))
    for node in assignments:
        target_names = [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ]
        if not target_names:
            continue
        try:
            value = _eval_string_expr(node.value, env)
        except ValueError:
            continue
        for name in target_names:
            if name in names or name in {
                "dynamic_token_count_anchor",
                "dynamic_pos_count_anchor",
            }:
                env[name] = value
    return {name: env[name] for name in names if name in env}


def check_dynamic_position_counter_partial_upgrade(checks: list[dict[str, Any]], patch_root: Path) -> None:
    worker = patch_root / "nemo_rl" / "models" / "generation" / "vllm" / "vllm_worker.py"
    if not worker.exists():
        add(checks, "source", "dynamic position-counter partial upgrade", "fail", f"missing: {worker}")
        return
    text = read_text(worker)
    required_markers = [
        "NRL_SPECDEC_SCHEDULER_DYNAMIC_POS_COUNTERS_PARTIAL_UPGRADE_V1",
        "def _has_scheduler_dynamic_pos_counter_marker",
        "not _has_scheduler_dynamic_pos_counter_marker(",
        "dynamic_pos_count_anchor",
        "dynamic_pos_count_block",
    ]
    missing_markers = [marker for marker in required_markers if marker not in text]
    wanted = {
        "dynamic_token_count_anchor",
        "dynamic_token_count_block",
        "dynamic_pos_count_anchor",
        "dynamic_pos_count_block",
    }
    try:
        strings = _extract_string_assignments(text, wanted)
    except SyntaxError as exc:
        add(
            checks,
            "source",
            "dynamic position-counter partial upgrade",
            "fail",
            "could not parse vLLM worker overlay while checking partial-upgrade guard",
            error=str(exc),
        )
        return
    missing_assignments = sorted(wanted - strings.keys())

    problems = []
    if missing_markers:
        problems.append({"missing_markers": missing_markers})
    if missing_assignments:
        problems.append({"missing_assignments": missing_assignments})

    if not problems:
        clean_upgrade = strings["dynamic_token_count_anchor"].replace(
            strings["dynamic_token_count_anchor"],
            strings["dynamic_token_count_block"],
            1,
        )
        partial_upgrade = strings["dynamic_pos_count_anchor"].replace(
            strings["dynamic_pos_count_anchor"],
            strings["dynamic_pos_count_block"],
            1,
        )
        clean_expected = [
            "_nrl_specdec_scheduler_dynamic_small_selected_token_count",
            "_nrl_specdec_scheduler_dynamic_pos{_nrl_pos_idx}_selected_count",
            "range(1, 9)",
        ]
        partial_expected = [
            "_nrl_specdec_scheduler_dynamic_large_selected_token_count",
            "_nrl_specdec_scheduler_dynamic_pos{_nrl_pos_idx}_selected_count",
            "range(1, 9)",
        ]
        clean_missing = [
            snippet for snippet in clean_expected if snippet not in clean_upgrade
        ]
        if clean_missing:
            problems.append({"clean token-counter upgrade": clean_missing})
        partial_missing = [
            snippet for snippet in partial_expected if snippet not in partial_upgrade
        ]
        if partial_missing:
            problems.append({"partial position-counter repair": partial_missing})

    if problems:
        add(
            checks,
            "source",
            "dynamic position-counter partial upgrade",
            "fail",
            "partial scheduler upgrades may still miss dynamic per-position denominator counters",
            problems=problems,
        )
    else:
        add(
            checks,
            "source",
            "dynamic position-counter partial upgrade",
            "pass",
            "clean and token-counter-only partial scheduler upgrade snippets both restore dynamic per-position counters",
        )


def _load_scheduler_output_gate_helper(worker: Path):
    text = read_text(worker)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_scheduler_output_gate_attr":
            source = ast.get_source_segment(text, node)
            if source is None:
                break
            namespace: dict[str, Any] = {
                "re": re,
                "scheduler_output_gate_marker": "NRL_SPECDEC_SCHEDULER_OUTPUT_GATE_ATTR_V2",
            }
            exec(textwrap.dedent(source), namespace)
            return namespace["_ensure_scheduler_output_gate_attr"]
    raise RuntimeError("could not find _ensure_scheduler_output_gate_attr in vllm_worker.py")


def _compile_text(text: str, filename: str) -> None:
    with tempfile.TemporaryDirectory(prefix="nrl_specdec_validate_") as tmp:
        path = Path(tmp) / filename
        path.write_text(text, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)


def _scheduler_output_gate_fixture() -> str:
    return textwrap.dedent(
        """
        class Scheduler:
            def schedule(self):
                if self.connector is not None:
                    self.connector.before_output()

                # Spec decode-related.
                scheduled_spec_decode_tokens: dict[str, list[int]] = {}

                if True:
                    disabled = False
                    self._nrl_specdec_scheduler_gate_last_disabled = disabled

                scheduler_output = SchedulerOutput(
                    scheduled_new_reqs=[],
                )

                if self.connector is not None:
                    self.connector.after_output(scheduler_output)

                return scheduler_output
        """
    ).lstrip()


def _discover_vllm_scheduler_source() -> Path | None:
    env_candidates = [
        os.environ.get("NRL_VALIDATE_VLLM_SCHEDULER_SOURCE"),
        os.environ.get("VLLM_SCHEDULER_SOURCE"),
    ]
    for candidate in env_candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    source_roots = []
    for env_name in ("VLLM_SOURCE_DIR", "VLLM_SITE"):
        value = os.environ.get(env_name)
        if value:
            source_roots.append(Path(value))
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception:
        spec = None
    if spec and spec.submodule_search_locations:
        source_roots.extend(Path(location) for location in spec.submodule_search_locations)

    for root in source_roots:
        if root.name == "vllm":
            path = root / "v1" / "core" / "sched" / "scheduler.py"
        else:
            path = root / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
        if path.exists():
            return path
    return None


def check_runner_gate_first_draft_guard(
    checks: list[dict[str, Any]], patch_root: Path
) -> None:
    worker = patch_root / "nemo_rl" / "models" / "generation" / "vllm" / "vllm_worker.py"
    if not worker.exists():
        add(checks, "source", "runner first-draft gate guard", "fail", f"missing: {worker}")
        return

    text = read_text(worker)
    required = [
        "NRL_SPECDEC_BATCH_GATE_PATCH_V7",
        "nrl_specdec_batch_gate_all_disabled",
        "nrl_specdec_batch_gate_eligible_count",
        "specdec_scheduler_all_attr",
        "_nrl_specdec_batch_gate_last_scheduler_eligible_count",
    ]
    missing = [snippet for snippet in required if snippet not in text]
    deadlock_snippet = (
        "specdec_batch_gate_disabled = (\\n\"\n"
        "                            \"                    specdec_batch_gate_num_requests > 0\\n\"\n"
        "                            \"                    and specdec_scheduled_token_count == 0\\n\"\n"
        "                            \"                )\\n\""
    )
    if missing or deadlock_snippet in text:
        add(
            checks,
            "source",
            "runner first-draft gate guard",
            "fail",
            "runner gate may still disable EAGLE before the first draft instead of using explicit scheduler all-disabled state",
            missing=missing,
            has_deadlock_snippet=deadlock_snippet in text,
        )
        return

    add(
        checks,
        "source",
        "runner first-draft gate guard",
        "pass",
        "runner gate uses explicit scheduler all-disabled state and no longer treats an empty scheduled_spec_decode_tokens map as a standalone disable signal",
    )


def check_scheduler_output_gate_runtime_fixture(checks: list[dict[str, Any]], patch_root: Path) -> None:
    worker = patch_root / "nemo_rl" / "models" / "generation" / "vllm" / "vllm_worker.py"
    if not worker.exists():
        add(checks, "source", "scheduler output gate runtime fixture", "fail", f"missing: {worker}")
        return

    try:
        helper = _load_scheduler_output_gate_helper(worker)
    except Exception as exc:
        add(
            checks,
            "source",
            "scheduler output gate runtime fixture",
            "fail",
            "could not load runtime scheduler-output gate helper from overlay",
            error=str(exc),
        )
        return

    marker = "NRL_SPECDEC_SCHEDULER_OUTPUT_GATE_ATTR_V2"
    scheduler_anchor = "        scheduler_output = SchedulerOutput(\n"
    try:
        fixture = _scheduler_output_gate_fixture()
        patched, changed = helper(fixture, Path("<synthetic_scheduler.py>"))
        repatched, changed_again = helper(patched, Path("<synthetic_scheduler.py>"))
        _compile_text(patched, "synthetic_scheduler.py")

        scheduler_pos = patched.find(scheduler_anchor)
        marker_pos = patched.find(marker)
        if not changed:
            raise AssertionError("helper did not report a change for an unpatched scheduler fixture")
        if changed_again or repatched != patched:
            raise AssertionError("helper is not idempotent on an already patched scheduler fixture")
        if scheduler_pos < 0:
            raise AssertionError("patched fixture lost the SchedulerOutput construction anchor")
        if marker_pos < 0:
            raise AssertionError("patched fixture is missing the scheduler-output gate marker")
        if marker in patched[:scheduler_pos]:
            raise AssertionError("scheduler-output gate marker was inserted before SchedulerOutput")
        required_runtime_attrs = [
            "nrl_specdec_batch_gate_disabled",
            "nrl_specdec_batch_gate_all_disabled",
            "nrl_specdec_batch_gate_eligible_count",
            "nrl_specdec_batch_gate_checked_count",
            "_nrl_specdec_scheduler_gate_output_all_disabled",
            "_nrl_specdec_scheduler_gate_output_checked",
            "_nrl_specdec_scheduler_gate_output_enabled",
        ]
        missing_runtime_attrs = [
            item for item in required_runtime_attrs if item not in patched
        ]
        if missing_runtime_attrs:
            raise AssertionError(
                "patched fixture is missing scheduler-output gate attrs: "
                + ", ".join(missing_runtime_attrs)
            )
    except Exception as exc:
        add(
            checks,
            "source",
            "scheduler output gate runtime fixture",
            "fail",
            "runtime scheduler-output gate helper failed synthetic insertion/idempotence/compile test",
            error=str(exc),
        )
        return

    add(
        checks,
        "source",
        "scheduler output gate runtime fixture",
        "pass",
        "runtime helper inserts scheduler_output gate state/all-disabled attrs after SchedulerOutput, is idempotent, and py_compiles",
    )

    scheduler_source = _discover_vllm_scheduler_source()
    if scheduler_source is None:
        add(
            checks,
            "source",
            "scheduler output gate exact source fixture",
            "warn",
            "no vLLM scheduler.py source was visible; set NRL_VALIDATE_VLLM_SCHEDULER_SOURCE to validate an exact generated source file",
        )
        return

    try:
        source_text = read_text(scheduler_source)
        patched_source, _ = helper(source_text, scheduler_source)
        _compile_text(patched_source, "scheduler.py")
        scheduler_pos = patched_source.find(scheduler_anchor)
        marker_pos = patched_source.find(marker)
        if scheduler_pos >= 0 and marker_pos >= 0 and marker in patched_source[:scheduler_pos]:
            raise AssertionError("scheduler-output gate marker was inserted before SchedulerOutput")
    except Exception as exc:
        add(
            checks,
            "source",
            "scheduler output gate exact source fixture",
            "warn",
            "visible vLLM scheduler.py could not be validated directly; it may be an unpatched upstream source missing earlier gate anchors",
            path=str(scheduler_source),
            error=str(exc),
        )
    else:
        add(
            checks,
            "source",
            "scheduler output gate exact source fixture",
            "pass",
            "runtime helper also py_compiles on the visible vLLM scheduler.py source",
            path=str(scheduler_source),
        )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    patch_root = args.patch_root
    target_root = args.target_specdec_rl_dir
    files, ignored = iter_overlay_files(patch_root)
    rows = required_file_rows(patch_root, target_root)

    if patch_root.exists():
        add(checks, "source", "patch root", "pass", f"visible: {patch_root}")
    else:
        add(checks, "source", "patch root", "fail", f"not visible: {patch_root}")

    for row in rows:
        rel = row["path"]
        if row["source_status"] == "pass":
            add(checks, "source", rel, "pass", "required overlay file contains expected markers")
        else:
            add(
                checks,
                "source",
                rel,
                "fail",
                "required overlay file is missing or lacks expected markers",
                missing_snippets=row.get("source_missing_snippets"),
            )

    check_grpo_generation_indent(checks, patch_root)
    check_python_sources_compile(checks, patch_root)
    check_dynamic_request_id_arity_guard(checks, patch_root)
    check_dynamic_position_counter_partial_upgrade(checks, patch_root)
    check_runner_gate_first_draft_guard(checks, patch_root)
    check_scheduler_output_gate_runtime_fixture(checks, patch_root)

    if ignored:
        add(
            checks,
            "source",
            "ignored generated files",
            "warn",
            "generated or binary files are present but excluded from the handoff overlay",
            count=len(ignored),
            examples=[str(path.relative_to(patch_root)) for path in ignored[:8]],
        )
    else:
        add(checks, "source", "ignored generated files", "pass", "no generated files found in the overlay tree")

    if target_root.exists():
        add(checks, "target", "SpecDec-RL checkout", "pass", f"visible: {target_root}")
        target_needs = [row for row in rows if row.get("target_status") != "applied"]
        if target_needs:
            status = "fail" if args.require_target_applied else "warn"
            add(
                checks,
                "target",
                "overlay applied",
                status,
                "target checkout is visible but one or more required files do not contain the overlay markers",
                missing=[row["path"] for row in target_needs],
            )
        else:
            add(checks, "target", "overlay applied", "pass", "target checkout contains all required overlay markers")
    else:
        status = "fail" if args.require_target_applied else "warn"
        add(checks, "target", "SpecDec-RL checkout", status, f"not visible: {target_root}")

    status_counts: dict[str, int] = {}
    for check in checks:
        status_counts[check["status"]] = status_counts.get(check["status"], 0) + 1

    source_failures = [
        check
        for check in checks
        if check.get("area") == "source" and check.get("status") == "fail"
    ]
    source_ok = (
        all(row.get("source_status") == "pass" for row in rows)
        and patch_root.exists()
        and not source_failures
    )
    target_ok = target_root.exists() and all(row.get("target_status") == "applied" for row in rows)
    if not source_ok:
        overall = "fail"
    elif args.require_target_applied and not target_ok:
        overall = "fail"
    else:
        overall = "pass"

    copy_command = (
        f'rsync -a --exclude "__pycache__" --exclude "*.pyc" '
        f'{patch_root.as_posix().rstrip("/")}/ {target_root.as_posix().rstrip("/")}/'
    )
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "patch_root": str(patch_root),
        "target_specdec_rl_dir": str(target_root),
        "target_status": "applied" if target_ok else ("not_visible" if not target_root.exists() else "needs_overlay"),
        "file_count": len(files),
        "ignored_file_count": len(ignored),
        "required_files": rows,
        "copy_command": copy_command,
        "counts": status_counts,
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SpecDec-RL Remote Patch Bundle Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Patch root: `{payload['patch_root']}`",
        f"Target checkout: `{payload['target_specdec_rl_dir']}`",
        f"Target status: `{payload['target_status']}`",
        "",
        "## Required Files",
        "",
        "| path | source | target |",
        "| --- | --- | --- |",
    ]
    for row in payload["required_files"]:
        lines.append(f"| `{row['path']}` | {row.get('source_status')} | {row.get('target_status')} |")
    lines += [
        "",
        "## Stage Command",
        "",
        "```bash",
        payload["copy_command"],
        "```",
        "",
        "## Checks",
        "",
        "| area | check | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {check['area']} | {check['name']} | {check['status'].upper()} | {detail} |")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    markdown = render_markdown(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 1 if payload["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
