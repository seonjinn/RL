#!/usr/bin/env python3
"""Export local TensorBoard scalars into the CUDA-graph result JSONL schema."""

import argparse
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

from collect_results import OPTIONAL_METRIC_MAP, WANDB_METRIC_MAP


EXPECTED_STEPS = frozenset(range(1, 21))

_PERFORMANCE_TAGS = tuple(WANDB_METRIC_MAP.values())
_CORRECTNESS_TAGS = (
    "train/reward",
    "train/token_mult_prob_error",
    "train/loss",
    OPTIONAL_METRIC_MAP["grad_norm"],
)
CANONICAL_TAG_ALIASES = {
    tag: (tag,) for tag in (*_PERFORMANCE_TAGS, *_CORRECTNESS_TAGS)
}
CANONICAL_TAG_ALIASES["train/reward"] = ("train/reward", "train/accuracy")


def _scalar_events(paths: Sequence[Path]) -> dict[str, dict[int, float]]:
    """Return latest scalar values per tag and optimizer step from local events."""
    values: dict[str, dict[int, tuple[float, int, int, float]]] = {}
    for source_index, path in enumerate(paths):
        accumulator = event_accumulator.EventAccumulator(
            str(path), size_guidance={event_accumulator.SCALARS: 0}
        )
        accumulator.Reload()
        for tag in accumulator.Tags().get(event_accumulator.SCALARS, []):
            for event_index, event in enumerate(accumulator.Scalars(tag)):
                rank = (event.wall_time, source_index, event_index)
                current = values.setdefault(tag, {}).get(event.step)
                if current is None or rank >= current[:3]:
                    values[tag][event.step] = (*rank, float(event.value))
    return {
        tag: {step: stored[-1] for step, stored in by_step.items()}
        for tag, by_step in values.items()
    }


def _canonical_metrics(
    scalar_values: Mapping[str, Mapping[int, float]],
) -> dict[int, dict[str, float]]:
    """Normalize required scalar tags and reject incomplete event exports."""
    rows = {step: {} for step in EXPECTED_STEPS}
    errors = []
    for canonical_tag, aliases in CANONICAL_TAG_ALIASES.items():
        for step in EXPECTED_STEPS:
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
        missing_steps = sorted(EXPECTED_STEPS - present_steps)
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
        raise ValueError("incomplete TensorBoard metrics: " + "; ".join(errors))
    return rows


def _write_jsonl_atomic(records: Iterable[Mapping[str, object]], output: Path) -> None:
    """Publish complete JSONL with replace semantics, never a partial file."""
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
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def export_events(
    event_paths: Sequence[Path],
    *,
    scope: str,
    job_id: str,
    status: str,
    output: Path,
) -> None:
    """Export exactly twenty complete optimizer steps from local event files."""
    if not event_paths:
        raise ValueError("at least one TensorBoard event path is required")
    missing_paths = [str(path) for path in event_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "missing TensorBoard event paths: " + ", ".join(missing_paths)
        )

    metrics_by_step = _canonical_metrics(_scalar_events(event_paths))
    _write_jsonl_atomic(
        (
            {
                "scope": scope,
                "job_id": job_id,
                "status": status,
                "step": step,
                "metrics": metrics_by_step[step],
            }
            for step in sorted(EXPECTED_STEPS)
        ),
        output,
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
    parser.add_argument("--scope", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the local event exporter."""
    args = parse_args()
    export_events(
        args.event,
        scope=args.scope,
        job_id=args.job_id,
        status=args.status,
        output=args.output,
    )


if __name__ == "__main__":
    main()
