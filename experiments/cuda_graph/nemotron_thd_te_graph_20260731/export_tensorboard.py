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

"""Export complete local TensorBoard scalars into experiment JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PLANNED_STEP_COUNTS = frozenset({5, 20, 100})
IDENTITY_FIELDS = (
    "model",
    "dispatcher",
    "scope",
    "mode",
    "cluster",
    "profile",
    "phase",
    "steps",
    "repeat",
    "run_group",
    "job_id",
    "router_replay",
)
INTEGER_IDENTITY_FIELDS = frozenset({"steps", "repeat"})
METADATA_PROVENANCE_FIELDS = {
    "nemo_rl_commit": "nemo_rl_commit",
    "bridge_commit": "bridge_commit",
    "mcore_commit": "mcore_commit",
    "transformer_engine_commit": "te_commit",
    "container_sha256": "container_sha256",
}
PROVENANCE_FIELDS = (
    "nemo_rl_commit",
    "bridge_commit",
    "mcore_commit",
    "te_commit",
    "te_version",
    "container_sha256",
)
COMMIT_FIELDS = frozenset(
    {"nemo_rl_commit", "bridge_commit", "mcore_commit", "te_commit"}
)
PARITY_FIELDS = frozenset(
    {
        "router_topk_parity",
        "expert_count_parity",
        "parameter_delta_parity",
        "parameter_delta_max_abs_error",
        "parameter_delta_max_rel_error",
    }
)
CANONICAL_TAG_ALIASES: dict[str, tuple[str, ...]] = {
    "timing/train/total_step_time": ("timing/train/total_step_time",),
    "timing/train/generation": ("timing/train/generation",),
    "timing/train/policy_training": ("timing/train/policy_training",),
    "timing/train/policy_and_reference_logprobs": (
        "timing/train/policy_and_reference_logprobs",
    ),
    "performance/tokens_per_sec_per_gpu": ("performance/tokens_per_sec_per_gpu",),
    "performance/generation_tokens_per_sec_per_gpu": (
        "performance/generation_tokens_per_sec_per_gpu",
    ),
    "performance/policy_training_tokens_per_sec_per_gpu": (
        "performance/policy_training_tokens_per_sec_per_gpu",
    ),
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    ),
    "cuda_graph/capture_count": (
        "train/cuda_graph/capture_count",
        "cuda_graph/capture_count",
    ),
    "cuda_graph/replay_count": (
        "train/cuda_graph/replay_count",
        "cuda_graph/replay_count",
    ),
    "cuda_graph/cache_hits": (
        "train/cuda_graph/cache_hits",
        "train/cuda_graph/cache_hit_count",
        "cuda_graph/cache_hits",
        "cuda_graph/cache_hit_count",
    ),
    "cuda_graph/cache_misses": (
        "train/cuda_graph/cache_misses",
        "train/cuda_graph/cache_miss_count",
        "cuda_graph/cache_misses",
        "cuda_graph/cache_miss_count",
    ),
    "cuda_graph/cache_evictions": (
        "train/cuda_graph/cache_evictions",
        "train/cuda_graph/eviction_count",
        "cuda_graph/cache_evictions",
        "cuda_graph/eviction_count",
    ),
    "cuda_graph/fallback_count": (
        "train/cuda_graph/fallback_count",
        "cuda_graph/fallback_count",
    ),
    "cuda_graph/graph_calls": (
        "train/cuda_graph/graph_calls",
        "cuda_graph/graph_calls",
    ),
    "cuda_graph/eligible_calls": (
        "train/cuda_graph/eligible_calls",
        "cuda_graph/eligible_calls",
    ),
    "cuda_graph/logical_tokens": (
        "train/cuda_graph/logical_tokens",
        "cuda_graph/logical_tokens",
    ),
    "cuda_graph/padded_tokens": (
        "train/cuda_graph/padded_tokens",
        "cuda_graph/padded_tokens",
    ),
    "cuda_graph/capacity_tokens": (
        "train/cuda_graph/capacity_tokens",
        "cuda_graph/capacity_tokens",
    ),
    "cuda_graph/coverage": (
        "train/cuda_graph/coverage",
        "cuda_graph/coverage",
    ),
    "cuda_graph/capacity_utilization": (
        "train/cuda_graph/capacity_utilization",
        "cuda_graph/capacity_utilization",
    ),
    "cuda_graph/padding_utilization": (
        "train/cuda_graph/padding_utilization",
        "cuda_graph/padding_utilization",
    ),
    "train/reward": ("train/reward", "train/accuracy"),
    "train/gen_kl_error": ("train/gen_kl_error",),
    "train/token_mult_prob_error": ("train/token_mult_prob_error",),
    "train/policy_kl_error": ("train/policy_kl_error",),
    "train/js_divergence_error": ("train/js_divergence_error",),
    "train/sampling_importance_ratio": ("train/sampling_importance_ratio",),
    "train/num_masked_seqs_by_logprob_error": (
        "train/num_masked_seqs_by_logprob_error",
        "train/num_mask_sample_filtered",
    ),
    "train/loss": ("train/loss",),
    "train/grad_norm": ("train/grad_norm",),
}
GRAPH_CANONICAL_TAGS = frozenset(
    tag for tag in CANONICAL_TAG_ALIASES if tag.startswith("cuda_graph/")
)
BASELINE_SCOPES = frozenset({"baseline", "baseline_no_cg"})
ROUTER_REPLAY_VALUES = frozenset({"off", "on"})


@dataclass(frozen=True)
class RunIdentity:
    """Canonical launch identity shared by every metric source."""

    model: str
    dispatcher: str
    scope: str
    mode: str
    cluster: str
    profile: str
    phase: str
    steps: int
    repeat: int
    run_group: str
    job_id: str
    router_replay: str


def _read_run_metadata(path: Path) -> dict[str, str]:
    """Read literal run metadata without treating it as shell input."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError(f"run metadata is not a safe regular file: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"run metadata is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 65536):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        content = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("run metadata must be UTF-8") from error
    if "\x00" in content or "\r" in content:
        raise ValueError("run metadata contains forbidden bytes")
    values: dict[str, str] = {}
    forbidden = frozenset("\\\"'`$;|&<>")
    for line_number, line in enumerate(content.splitlines(), start=1):
        if (
            not line
            or any(character.isspace() for character in line)
            or any(character in forbidden for character in line)
        ):
            raise ValueError(f"run metadata line {line_number} is unsafe")
        if "=" not in line:
            raise ValueError(f"run metadata line {line_number} is malformed")
        key, value = line.split("=", 1)
        if re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
            raise ValueError(f"run metadata line {line_number} has invalid key")
        if key in values:
            raise ValueError(f"run metadata duplicates {key}")
        values[key] = value
    if not values:
        raise ValueError("run metadata must contain canonical assignments")
    return values


def _metadata_identity(metadata: Mapping[str, str]) -> dict[str, object]:
    missing = [field for field in IDENTITY_FIELDS if field not in metadata]
    if missing:
        raise ValueError("run metadata identity is missing: " + ", ".join(missing))
    normalized: dict[str, object] = {
        field: metadata[field] for field in IDENTITY_FIELDS
    }
    for field in ("steps", "repeat"):
        value = str(normalized[field])
        if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
            raise ValueError(f"run metadata {field} must be a non-negative integer")
        normalized[field] = int(value)
    if normalized["router_replay"] not in ROUTER_REPLAY_VALUES:
        raise ValueError("run metadata router_replay must be one of off, on")
    return normalized


def _normalize_explicit_integer(field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _resolve_identity(
    *,
    metadata: Mapping[str, str] | None,
    model: str | None,
    dispatcher: str | None,
    scope: str | None,
    mode: str | None,
    cluster: str | None,
    profile: str | None,
    phase: str | None,
    steps: int | None,
    repeat: int | None,
    run_group: str | None,
    job_id: str | None,
    router_replay: str | None,
) -> RunIdentity:
    explicit: dict[str, object | None] = {
        "model": model,
        "dispatcher": dispatcher,
        "scope": scope,
        "mode": mode,
        "cluster": cluster,
        "profile": profile,
        "phase": phase,
        "steps": steps,
        "repeat": repeat,
        "run_group": run_group,
        "job_id": job_id,
        "router_replay": router_replay,
    }
    for field in INTEGER_IDENTITY_FIELDS:
        if explicit[field] is not None:
            explicit[field] = _normalize_explicit_integer(field, explicit[field])

    resolved = _metadata_identity(metadata) if metadata is not None else {}
    for field in IDENTITY_FIELDS:
        value = explicit[field]
        if value is not None and field in resolved and value != resolved[field]:
            raise ValueError(f"run metadata {field} disagrees with explicit identity")
        if value is not None:
            resolved[field] = value

    missing = [field for field in IDENTITY_FIELDS if field not in resolved]
    if missing:
        raise ValueError("launch identity is missing: " + ", ".join(missing))
    for field in set(IDENTITY_FIELDS) - INTEGER_IDENTITY_FIELDS:
        if not isinstance(resolved[field], str) or not resolved[field]:
            raise ValueError(f"{field} must be a non-empty string")
    if resolved["steps"] not in PLANNED_STEP_COUNTS:
        raise ValueError("steps must be one of 5, 20, 100")
    if resolved["mode"] not in {"nemorl", "mcore"}:
        raise ValueError("mode must be one of nemorl, mcore")
    if resolved["router_replay"] not in ROUTER_REPLAY_VALUES:
        raise ValueError("router_replay must be one of off, on")
    if not str(resolved["profile"]).strip():
        raise ValueError("profile must not be empty")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(resolved["run_group"])) is None:
        raise ValueError("run_group must be filesystem-safe")
    if re.fullmatch(r"[1-9][0-9]*", str(resolved["job_id"])) is None:
        raise ValueError("job_id must be a positive decimal identifier")

    return RunIdentity(
        model=str(resolved["model"]),
        dispatcher=str(resolved["dispatcher"]),
        scope=str(resolved["scope"]),
        mode=str(resolved["mode"]),
        cluster=str(resolved["cluster"]),
        profile=str(resolved["profile"]),
        phase=str(resolved["phase"]),
        steps=int(resolved["steps"]),
        repeat=int(resolved["repeat"]),
        run_group=str(resolved["run_group"]),
        job_id=str(resolved["job_id"]),
        router_replay=str(resolved["router_replay"]),
    )


def _scalar_events(paths: Sequence[Path]) -> dict[str, dict[int, float]]:
    """Return the latest scalar value per source tag and optimizer step."""
    from tensorboard.backend.event_processing import event_accumulator  # pyright: ignore[reportMissingImports]

    values: dict[str, dict[int, tuple[float, int, int, float]]] = {}
    for source_index, path in enumerate(paths):
        accumulator = event_accumulator.EventAccumulator(
            str(path), size_guidance={event_accumulator.SCALARS: 0}
        )
        accumulator.Reload()
        for tag in accumulator.Tags().get(event_accumulator.SCALARS, []):
            for event_index, event in enumerate(accumulator.Scalars(tag)):
                rank = (float(event.wall_time), source_index, event_index)
                current = values.setdefault(tag, {}).get(event.step)
                if current is None or rank >= current[:3]:
                    values[tag][event.step] = (*rank, float(event.value))
    return {
        tag: {step: stored[-1] for step, stored in by_step.items()}
        for tag, by_step in values.items()
    }


def _canonical_metrics(
    scalar_values: Mapping[str, Mapping[int, float]],
    *,
    steps: int,
    require_graph_metrics: bool = True,
) -> dict[int, dict[str, float]]:
    """Normalize aliases and reject missing or non-finite required values."""
    expected_steps = frozenset(range(1, steps + 1))
    rows = {step: {} for step in expected_steps}
    errors: list[str] = []
    for canonical_tag, aliases in CANONICAL_TAG_ALIASES.items():
        if canonical_tag in GRAPH_CANONICAL_TAGS and not require_graph_metrics:
            continue
        for step in expected_steps:
            value = next(
                (
                    scalar_values[alias][step]
                    for alias in aliases
                    if step in scalar_values.get(alias, {})
                ),
                None,
            )
            if value is not None:
                rows[step][canonical_tag] = float(value)

        present_steps = {
            step for step, metrics in rows.items() if canonical_tag in metrics
        }
        missing_steps = sorted(expected_steps - present_steps)
        invalid_steps = sorted(
            step
            for step in present_steps
            if not math.isfinite(rows[step][canonical_tag])
        )
        if missing_steps or invalid_steps:
            details = [
                f"missing_steps={missing_steps}",
                f"count={len(present_steps)}",
            ]
            if invalid_steps:
                details.append(f"invalid_steps={invalid_steps}")
            errors.append(f"{canonical_tag}: {', '.join(details)}")

    if errors:
        raise ValueError("incomplete metrics: " + "; ".join(errors))
    return rows


def _write_jsonl_atomic(records: Iterable[Mapping[str, object]], output: Path) -> None:
    """Replace the destination only after the complete JSONL is durable."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            for record in records:
                json.dump(record, temporary, allow_nan=False, separators=(",", ":"))
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        output.chmod(0o644)
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def _validated_provenance(provenance: Mapping[str, object]) -> dict[str, str]:
    """Return canonical, exact source/runtime provenance."""
    missing = [field for field in PROVENANCE_FIELDS if field not in provenance]
    if missing:
        raise ValueError("provenance is missing: " + ", ".join(missing))
    normalized = {field: str(provenance[field]) for field in PROVENANCE_FIELDS}
    for field in COMMIT_FIELDS:
        if re.fullmatch(r"[0-9a-f]{40}", normalized[field]) is None:
            raise ValueError(f"provenance.{field} must be a full lowercase commit SHA")
    if re.fullmatch(r"[0-9a-f]{64}", normalized["container_sha256"]) is None:
        raise ValueError(
            "provenance.container_sha256 must be a full lowercase SHA256 digest"
        )
    if re.fullmatch(r"\d+\.\d+(?:[A-Za-z0-9.+-]*)", normalized["te_version"]) is None:
        raise ValueError("provenance.te_version must be an exact version string")
    return normalized


def _validated_parity(parity: Mapping[str, object] | None) -> dict[str, object]:
    """Validate optional parity evidence without inventing missing values."""
    if parity is None:
        return {}
    unknown = sorted(set(parity) - PARITY_FIELDS)
    if unknown:
        raise ValueError("unsupported parity fields: " + ", ".join(unknown))
    normalized = dict(parity)
    for field in (
        "router_topk_parity",
        "expert_count_parity",
        "parameter_delta_parity",
    ):
        if field in normalized and type(normalized[field]) is not bool:
            raise TypeError(f"parity.{field} must be a bool")
    for field in (
        "parameter_delta_max_abs_error",
        "parameter_delta_max_rel_error",
    ):
        if field not in normalized:
            continue
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"parity.{field} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"parity.{field} must be finite and nonnegative")
        normalized[field] = numeric
    return normalized


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def resolve_export_context(
    *,
    model: str | None = None,
    dispatcher: str | None = None,
    scope: str | None = None,
    mode: str | None = None,
    cluster: str | None = None,
    profile: str | None = None,
    phase: str | None = None,
    steps: int | None = None,
    repeat: int | None = None,
    run_group: str | None = None,
    job_id: str | None = None,
    router_replay: str | None = None,
    run_metadata: Path | None = None,
    provenance: Mapping[str, object],
    parity: Mapping[str, object] | None,
) -> tuple[RunIdentity, dict[str, str], dict[str, object]]:
    """Resolve and cross-check identity before any metric source is read."""
    metadata = _read_run_metadata(run_metadata) if run_metadata is not None else None
    identity = _resolve_identity(
        metadata=metadata,
        model=model,
        dispatcher=dispatcher,
        scope=scope,
        mode=mode,
        cluster=cluster,
        profile=profile,
        phase=phase,
        steps=steps,
        repeat=repeat,
        run_group=run_group,
        job_id=job_id,
        router_replay=router_replay,
    )
    normalized_provenance = _validated_provenance(provenance)
    if metadata is not None:
        missing = sorted(set(METADATA_PROVENANCE_FIELDS) - set(metadata))
        if missing:
            raise ValueError(
                "run metadata provenance is missing: " + ", ".join(missing)
            )
        for metadata_field, provenance_field in METADATA_PROVENANCE_FIELDS.items():
            if metadata[metadata_field] != normalized_provenance[provenance_field]:
                raise ValueError(
                    f"run metadata {metadata_field} disagrees with provenance"
                )
    return identity, normalized_provenance, _validated_parity(parity)


def export_scalar_values(
    scalar_values: Mapping[str, Mapping[int, float]],
    *,
    identity: RunIdentity,
    status: str,
    provenance: Mapping[str, str],
    parity: Mapping[str, object],
    output: Path,
) -> None:
    """Export one already-resolved scalar source through the canonical schema."""
    is_baseline = identity.scope in BASELINE_SCOPES
    metrics_by_step = _canonical_metrics(
        scalar_values,
        steps=identity.steps,
        require_graph_metrics=not is_baseline,
    )
    common = {
        **asdict(identity),
        "status": status,
        "graph_telemetry_status": "not_applicable" if is_baseline else "reported",
        "provenance": dict(provenance),
        "parity": dict(parity),
    }
    _write_jsonl_atomic(
        (
            {**common, "step": step, "metrics": metrics_by_step[step]}
            for step in range(1, identity.steps + 1)
        ),
        output,
    )


def export_events(
    event_paths: Sequence[Path],
    *,
    model: str | None = None,
    dispatcher: str | None = None,
    scope: str | None = None,
    mode: str | None = None,
    cluster: str | None = None,
    profile: str | None = None,
    phase: str | None = None,
    steps: int | None = None,
    repeat: int | None = None,
    run_group: str | None = None,
    job_id: str | None = None,
    status: str,
    provenance: Mapping[str, object],
    output: Path,
    parity: Mapping[str, object] | None = None,
    router_replay: str | None = None,
    run_metadata: Path | None = None,
) -> None:
    """Export one complete planned run from local TensorBoard event paths."""
    identity, normalized_provenance, normalized_parity = resolve_export_context(
        model=model,
        dispatcher=dispatcher,
        scope=scope,
        mode=mode,
        cluster=cluster,
        profile=profile,
        phase=phase,
        steps=steps,
        repeat=repeat,
        run_group=run_group,
        job_id=job_id,
        router_replay=router_replay,
        run_metadata=run_metadata,
        provenance=provenance,
        parity=parity,
    )
    if not event_paths:
        raise ValueError("at least one TensorBoard event path is required")
    missing_paths = [str(path) for path in event_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "missing TensorBoard event paths: " + ", ".join(missing_paths)
        )

    export_scalar_values(
        _scalar_events(event_paths),
        identity=identity,
        status=status,
        provenance=normalized_provenance,
        parity=normalized_parity,
        output=output,
    )


def parse_args() -> argparse.Namespace:
    """Parse local-only TensorBoard export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event",
        action="append",
        required=True,
        type=Path,
        help="TensorBoard event file or directory; repeat for separate sources",
    )
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--dispatcher")
    parser.add_argument("--scope")
    parser.add_argument("--mode", choices=("nemorl", "mcore"))
    parser.add_argument("--cluster")
    parser.add_argument("--profile")
    parser.add_argument("--phase")
    parser.add_argument("--steps", choices=sorted(PLANNED_STEP_COUNTS), type=int)
    parser.add_argument("--job-id")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-group")
    parser.add_argument("--status", required=True)
    parser.add_argument("--router-replay", choices=("off", "on"))
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the TensorBoard exporter."""
    args = parse_args()
    export_events(
        args.event,
        model=args.model,
        dispatcher=args.dispatcher,
        scope=args.scope,
        mode=args.mode,
        cluster=args.cluster,
        profile=args.profile,
        phase=args.phase,
        steps=args.steps,
        repeat=args.repeat,
        run_group=args.run_group,
        job_id=args.job_id,
        status=args.status,
        router_replay=args.router_replay,
        run_metadata=args.run_metadata,
        provenance=_read_json_mapping(args.provenance, label="provenance"),
        parity=(
            _read_json_mapping(args.parity, label="parity")
            if args.parity is not None
            else None
        ),
        output=args.output,
    )


if __name__ == "__main__":
    main()
