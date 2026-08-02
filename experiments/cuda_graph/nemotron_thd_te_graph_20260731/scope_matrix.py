#!/usr/bin/env python3
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

"""Build, classify, and render the persistent CUDA Graph scope matrix."""

from __future__ import annotations

import argparse
import itertools
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
DENSE_AXES = ("attn", "mlp", "mamba")
MOE_AXES = (
    (),
    ("moe",),
    ("moe_router",),
    ("moe_router", "moe_preprocess"),
)
VALID_STEPS = (5, 20, 100)
MODEL_NAMES = ("nano", "super", "ultra", "qwen3_30ba3b", "qwen3_235b")
ALLOWED_MCORE_DRIVERS: frozenset[str] = frozenset()
Status = Literal[
    "runnable",
    "model-incompatible",
    "capacity-blocked",
    "dependency-blocked",
    "submitted",
]


@dataclass(frozen=True)
class ScopeRow:
    """One persistent scope-matrix row."""

    index: int
    name: str
    scope: tuple[str, ...]
    cuda_graph_enabled: bool


@dataclass(frozen=True)
class ModelSpec:
    """Model capabilities loaded from one committed selector."""

    name: str
    nemorl_launcher: str
    nemorl_launcher_validated: bool
    nemorl_recipe: str
    mcore_recipe: str
    dispatcher: str
    supported_modules: frozenset[str]
    whole_moe_capacity_ready: bool
    moe_preprocess_graph_ready: bool
    requires_ultra_externals: bool
    num_nodes: int
    gpus_per_node: int
    thd_max_packed_sequences: int


@dataclass(frozen=True)
class ScopeClassification:
    """Pre-submission status and its actionable explanation."""

    status: Status
    reason: str


def load_scope_matrix() -> tuple[ScopeRow, ...]:
    """Return one baseline followed by the exact 32 TE scope combinations."""
    rows = [ScopeRow(0, "baseline_no_cg", (), False)]
    index = 1
    for dense_enabled in itertools.product((False, True), repeat=3):
        dense_scope = tuple(
            module
            for enabled, module in zip(dense_enabled, DENSE_AXES, strict=True)
            if enabled
        )
        for moe_scope in MOE_AXES:
            scope = dense_scope + moe_scope
            name = "whole_layer" if not scope else "_".join(scope)
            name = name.replace("moe_router_moe_preprocess", "moe_router_preprocess")
            rows.append(ScopeRow(index, name, scope, True))
            index += 1
    return tuple(rows)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected NAME=value")
        name, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"{path}:{line_number}: invalid variable name {name!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def _required(values: Mapping[str, str], name: str, path: Path) -> str:
    try:
        value = values[name]
    except KeyError as error:
        raise ValueError(f"{path}: missing required selector field {name}") from error
    if not value:
        raise ValueError(f"{path}: selector field {name} must not be empty")
    return value


def _selector_bool(values: Mapping[str, str], name: str, path: Path) -> bool:
    value = _required(values, name, path)
    if value not in ("true", "false"):
        raise ValueError(f"{path}: {name} must be true or false")
    return value == "true"


def load_model_spec(model: str) -> ModelSpec:
    """Load one selector without executing shell code."""
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}")
    path = EXPERIMENT_DIR / "models" / f"{model}.env"
    values = _parse_env_file(path)
    supported_modules = frozenset(
        item for item in _required(values, "SUPPORTED_MODULES", path).split(",") if item
    )
    return ModelSpec(
        name=model,
        nemorl_launcher=_required(values, "NEMORL_LAUNCHER", path),
        nemorl_launcher_validated=_selector_bool(
            values, "NEMORL_LAUNCHER_VALIDATED", path
        ),
        nemorl_recipe=_required(values, "NEMORL_RECIPE", path),
        mcore_recipe=_required(values, "MCORE_RECIPE", path),
        dispatcher=_required(values, "DISPATCHER", path),
        supported_modules=supported_modules,
        whole_moe_capacity_ready=_selector_bool(
            values, "WHOLE_MOE_CAPACITY_READY", path
        ),
        moe_preprocess_graph_ready=_selector_bool(
            values, "MOE_PREPROCESS_GRAPH_READY", path
        ),
        requires_ultra_externals=_selector_bool(
            values, "REQUIRES_ULTRA_EXTERNALS", path
        ),
        num_nodes=int(_required(values, "NUM_NODES", path)),
        gpus_per_node=int(_required(values, "GPUS_PER_NODE", path)),
        thd_max_packed_sequences=int(
            _required(values, "THD_MAX_PACKED_SEQUENCES", path)
        ),
    )


def find_scope_row(value: str) -> ScopeRow:
    """Resolve the public baseline/whole-layer/comma-list scope syntax."""
    if value == "baseline":
        return load_scope_matrix()[0]
    normalized = () if value == "whole_layer" else tuple(value.split(","))
    for row in load_scope_matrix()[1:]:
        if row.scope == normalized:
            return row
    raise ValueError(f"unknown scope {value!r}")


def classify_scope(
    row: ScopeRow,
    *,
    model: str,
    mode: str = "nemorl",
    submitted_job_id: str | None = None,
    external_dependencies_ready: bool = False,
    mcore_driver: str | None = None,
    profile_ready: bool = True,
) -> ScopeClassification:
    """Classify a row before any scheduler call."""
    spec = load_model_spec(model)
    unsupported = tuple(
        module for module in row.scope if module not in spec.supported_modules
    )
    if unsupported:
        return ScopeClassification(
            "model-incompatible",
            f"{model} has no compatible modules: {','.join(unsupported)}",
        )
    requests_whole_moe = row.cuda_graph_enabled and (
        not row.scope or "moe" in row.scope
    )
    if requests_whole_moe and not spec.whole_moe_capacity_ready:
        return ScopeClassification(
            "capacity-blocked",
            "whole-MoE capture has no verified fixed drop-and-pad capacity",
        )
    if "moe_preprocess" in row.scope and not spec.moe_preprocess_graph_ready:
        return ScopeClassification(
            "capacity-blocked",
            "HybridEP moe_preprocess has no verified fixed-capacity geometry",
        )
    if not profile_ready:
        return ScopeClassification(
            "dependency-blocked", "cluster profile has unresolved runtime fields"
        )
    if mode not in ("nemorl", "mcore"):
        raise ValueError("mode must be nemorl or mcore")
    if (
        mode == "nemorl"
        and spec.requires_ultra_externals
        and not external_dependencies_ready
    ):
        return ScopeClassification(
            "dependency-blocked",
            "Ultra requires model path, data, judge config, and launch profile",
        )
    if mode == "nemorl" and not spec.nemorl_launcher_validated:
        return ScopeClassification(
            "dependency-blocked",
            f"{model} has no validated launcher adapter for {spec.nemorl_launcher}",
        )
    if mode == "mcore" and spec.mcore_recipe.startswith("__REQUIRED_"):
        return ScopeClassification(
            "dependency-blocked",
            f"{model} has no committed standalone MCore recipe",
        )
    if mode == "mcore" and not _is_committed_mcore_driver(mcore_driver):
        return ScopeClassification(
            "dependency-blocked",
            "MCORE_DRIVER must name an allowlisted committed standalone driver",
        )
    if submitted_job_id:
        return ScopeClassification("submitted", f"Slurm job {submitted_job_id}")
    return ScopeClassification("runnable", "all pre-submission checks passed")


def render_scope_command(
    *,
    model: str,
    scope: Sequence[str],
    steps: int,
    run_name: str,
    cuda_graph_enabled: bool = True,
    router_replay_enabled: bool = False,
    log_dir: str | None = None,
    extra_overrides: Sequence[str] = (),
) -> str:
    """Render one NeMo-RL driver command with the canonical graph fields."""
    spec = load_model_spec(model)
    if not spec.nemorl_launcher_validated:
        raise ValueError(
            f"{model} has no validated launcher adapter for {spec.nemorl_launcher}"
        )
    if steps not in VALID_STEPS:
        raise ValueError(f"steps must be one of {VALID_STEPS}, got {steps}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("run_name must be filesystem-safe")
    if spec.thd_max_packed_sequences < 2:
        raise ValueError("thd_max_packed_sequences must be at least 2")
    if router_replay_enabled and cuda_graph_enabled and {
        "moe_router",
        "moe_preprocess",
    }.intersection(scope):
        raise ValueError(
            "Router Replay cannot be combined with a router CUDA Graph scope"
        )
    protected_overrides = {"policy.router_replay.enabled"}
    if router_replay_enabled:
        protected_overrides.update(
            (
                "policy.generation.vllm_cfg.enable_prefix_caching",
                "policy.generation.vllm_kwargs.enable_chunked_prefill",
            )
        )
    for override in extra_overrides:
        override_key = override.split("=", 1)[0].lstrip("+")
        if override_key in protected_overrides:
            raise ValueError(f"protected Router Replay override: {override_key}")
    modules = ",".join(scope)
    command = [
        "env",
        "NRL_FORCE_REBUILD_VENVS=true",
        "uv",
        "run",
        spec.nemorl_launcher,
        "--config",
        spec.nemorl_recipe,
        f"grpo.max_num_steps={steps}",
        "checkpointing.enabled=false",
        "policy.sequence_packing.enabled=true",
        "policy.dynamic_batching.enabled=false",
        f"cluster.num_nodes={spec.num_nodes}",
        f"cluster.gpus_per_node={spec.gpus_per_node}",
        f"logger.log_dir={log_dir or f'exp_logs/nemotron_thd_te_graph_20260731/{run_name}'}",
        "logger.wandb_enabled=true",
        "logger.tensorboard_enabled=true",
        "logger.wandb.project=sna-cg-study",
        f"logger.wandb.name={run_name}",
        "++policy.router_replay.enabled="
        f"{str(router_replay_enabled).lower()}",
    ]
    if router_replay_enabled:
        command[2:2] = (
            "NRL_ROUTER_REPLAY_VALIDATE=1",
            "NRL_R3_TRACE=1",
            "NRL_R3_TRACE_STEPS=5",
            "NRL_R3_TRACE_VERIFY_FORWARD=1",
        )
        command.extend(
            (
                "++policy.generation.vllm_cfg.enable_prefix_caching=false",
                "++policy.generation.vllm_kwargs.enable_chunked_prefill=false",
            )
        )
    if spec.dispatcher == "hybridep":
        command.extend(
            (
                "policy.megatron_cfg.moe_token_dispatcher_type=flex",
                "++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep",
            )
        )
    elif spec.dispatcher in {"allgather", "alltoall", "alltoall_seq"}:
        command.append(
            f"policy.megatron_cfg.moe_token_dispatcher_type={spec.dispatcher}"
        )
    else:
        raise ValueError(f"unsupported dispatcher contract {spec.dispatcher!r}")
    if cuda_graph_enabled:
        command.extend(
            (
                "++policy.megatron_cfg.cuda_graph_impl=transformer_engine",
                f"++policy.megatron_cfg.cuda_graph_modules=[{modules}]",
                "++policy.megatron_cfg.cuda_graph_warmup_steps=3",
                "++policy.megatron_cfg.thd_max_packed_sequences="
                f"{spec.thd_max_packed_sequences}",
            )
        )
    else:
        command.append("++policy.megatron_cfg.cuda_graph_impl=none")
    command.extend(extra_overrides)
    return shlex.join(command)


def _is_committed_mcore_driver(driver: str | None) -> bool:
    """Reject arbitrary shell snippets until a reviewed driver is allowlisted."""
    return driver in ALLOWED_MCORE_DRIVERS if driver is not None else False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("list")

    classify = subparsers.add_parser("classify")
    classify.add_argument("--model", choices=MODEL_NAMES, required=True)
    classify.add_argument("--scope", required=True)
    classify.add_argument("--mode", choices=("nemorl", "mcore"), default="nemorl")
    classify.add_argument("--external-dependencies-ready", action="store_true")
    classify.add_argument("--mcore-driver")
    classify.add_argument("--profile-blocked", action="store_true")

    render = subparsers.add_parser("render")
    render.add_argument("--model", choices=MODEL_NAMES, required=True)
    render.add_argument("--scope", required=True)
    render.add_argument("--steps", type=int, choices=VALID_STEPS, required=True)
    render.add_argument("--run-name", required=True)
    render.add_argument("--log-dir")
    render.add_argument("--router-replay", choices=("off", "on"), default="off")
    render.add_argument("--override", action="append", default=[])
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.action == "list":
        for row in load_scope_matrix():
            scope = (
                "baseline"
                if not row.cuda_graph_enabled
                else (",".join(row.scope) if row.scope else "whole_layer")
            )
            print(
                f"{row.index:02d}\t{row.name}\t{scope}\t"
                f"cuda_graph_enabled={str(row.cuda_graph_enabled).lower()}"
            )
        return
    row = find_scope_row(args.scope)
    if args.action == "classify":
        classification = classify_scope(
            row,
            model=args.model,
            mode=args.mode,
            external_dependencies_ready=args.external_dependencies_ready,
            mcore_driver=args.mcore_driver,
            profile_ready=not args.profile_blocked,
        )
        print(f"{classification.status}\t{classification.reason}")
        return
    print(
        render_scope_command(
            model=args.model,
            scope=row.scope,
            steps=args.steps,
            run_name=args.run_name,
            cuda_graph_enabled=row.cuda_graph_enabled,
            router_replay_enabled=args.router_replay == "on",
            log_dir=args.log_dir,
            extra_overrides=args.override,
        )
    )


if __name__ == "__main__":
    main()
