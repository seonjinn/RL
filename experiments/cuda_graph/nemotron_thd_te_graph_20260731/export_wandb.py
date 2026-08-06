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

"""Export complete W&B history into the canonical experiment JSONL schema."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Protocol

from export_tensorboard import (
    CANONICAL_TAG_ALIASES,
    PLANNED_STEP_COUNTS,
    _read_json_mapping,
    export_scalar_values,
    resolve_export_context,
)


class HistoryRun(Protocol):
    """Minimal W&B public-run behavior needed by the pure exporter."""

    def scan_history(self) -> Iterable[Mapping[str, object]]:
        """Yield unfiltered history rows."""
        ...


def _validated_step(value: object, *, steps: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("optimizer step must be a finite integer")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("optimizer step must be a finite integer")
    step = int(numeric)
    if step < 1 or step > steps:
        raise ValueError(f"optimizer step {step} is outside planned steps 1..{steps}")
    return step


def _validated_metric(tag: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"metric {tag} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite metric {tag}")
    return numeric


def coalesce_history(
    rows: Iterable[Mapping[str, object]],
    *,
    optimizer_step_keys: Sequence[str],
    steps: int,
) -> dict[str, dict[int, float]]:
    """Coalesce partial raw-tag rows by an explicitly mapped optimizer step."""
    if isinstance(optimizer_step_keys, str) or not optimizer_step_keys:
        raise ValueError("at least one optimizer step key is required")
    if any(not isinstance(key, str) or not key for key in optimizer_step_keys):
        raise ValueError("optimizer step keys must be unique non-empty strings")
    if len(set(optimizer_step_keys)) != len(optimizer_step_keys):
        raise ValueError("optimizer step keys must be unique non-empty strings")

    source_tags = frozenset(
        alias for aliases in CANONICAL_TAG_ALIASES.values() for alias in aliases
    )
    if source_tags.intersection(optimizer_step_keys):
        raise ValueError("optimizer step keys must not be metric tags")

    values: dict[str, dict[int, float]] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"W&B history row {row_index} must be a mapping")
        metric_items = [
            (tag, row[tag])
            for tag in source_tags
            if tag in row and row[tag] is not None
        ]
        if not metric_items:
            continue
        step_values = [
            _validated_step(row[key], steps=steps)
            for key in optimizer_step_keys
            if key in row and row[key] is not None
        ]
        if not step_values:
            raise ValueError(f"W&B history row {row_index} is missing optimizer step")
        if len(set(step_values)) != 1:
            raise ValueError(
                f"W&B history row {row_index} has conflicting optimizer step identities"
            )
        step = step_values[0]
        for tag, value in metric_items:
            values.setdefault(tag, {})[step] = _validated_metric(tag, value)
    return values


def export_run(
    run: HistoryRun,
    *,
    optimizer_step_keys: Sequence[str],
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
    status: str,
    provenance: Mapping[str, object],
    parity: Mapping[str, object] | None,
    output: Path,
) -> None:
    """Export one W&B run without reading summaries or zero-filling rows."""
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
    scalar_values = coalesce_history(
        run.scan_history(),
        optimizer_step_keys=optimizer_step_keys,
        steps=identity.steps,
    )
    export_scalar_values(
        scalar_values,
        identity=identity,
        status=status,
        provenance=normalized_provenance,
        parity=normalized_parity,
        output=output,
    )


def parse_args() -> argparse.Namespace:
    """Parse W&B Public API export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-run", required=True, help="entity/project/run_id")
    parser.add_argument("--optimizer-step-key", action="append", required=True)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--dispatcher")
    parser.add_argument("--scope")
    parser.add_argument("--mode", choices=("nemorl", "mcore"))
    parser.add_argument("--cluster")
    parser.add_argument("--profile")
    parser.add_argument("--phase")
    parser.add_argument("--steps", choices=sorted(PLANNED_STEP_COUNTS), type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-group")
    parser.add_argument("--job-id")
    parser.add_argument("--router-replay", choices=("off", "on"))
    parser.add_argument("--status", required=True)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Retrieve one public W&B run and export its unfiltered history."""
    args = parse_args()
    import wandb  # pyright: ignore[reportMissingImports]

    run = wandb.Api().run(args.wandb_run)
    export_run(
        run,
        optimizer_step_keys=args.optimizer_step_key,
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
        router_replay=args.router_replay,
        run_metadata=args.run_metadata,
        status=args.status,
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
