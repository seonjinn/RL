"""Select fixed-K and DynamicSD policies from Q30 calibration results."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
import math
from typing import Iterable

from .contract import Drafter, ExperimentContract


THROUGHPUT_OBJECTIVE = (
    "median per-result output_tokens / elapsed_seconds for each batch/K cell; "
    "equal-weight mean of cell medians across batch sizes for fixed K"
)


@dataclass(frozen=True, slots=True)
class CalibrationResultRow:
    """One validated calibration result for a batch/K/repetition cell."""

    drafter: Drafter
    batch_size: int
    verifier_k: int
    repetition: int
    output_tokens: int
    elapsed_seconds: float
    validated: bool

    @property
    def output_tokens_per_second(self) -> float:
        """Return this result's natural-EOS throughput objective."""
        return self.output_tokens / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    """Calibrated fixed K and contiguous monotone DynamicSD schedule."""

    drafter: Drafter
    best_fixed_k: int
    schedule: list[list[int]]
    throughput_objective: str = THROUGHPUT_OBJECTIVE


def _validated_rows(
    rows: Iterable[CalibrationResultRow],
    *,
    drafter: Drafter,
) -> tuple[CalibrationResultRow, ...]:
    if drafter not in ("dflash", "dspark"):
        raise ValueError(f"unsupported drafter: {drafter!r}")
    candidates = tuple(rows)
    if not candidates:
        raise ValueError(
            "calibration results do not contain the exact required BS/K grid"
        )

    contract = ExperimentContract()
    capability = {"dflash": 7, "dspark": 8}[drafter]
    identities: set[tuple[int, int, int]] = set()
    for row in candidates:
        if type(row) is not CalibrationResultRow:
            raise ValueError(
                "calibration input must contain validated CalibrationResultRow records"
            )
        if type(row.validated) is not bool:
            raise ValueError("calibration result validated must be a boolean")
        if not row.validated:
            raise ValueError("unvalidated calibration result is not selectable")
        if row.drafter != drafter:
            raise ValueError("calibration results must contain one requested drafter")
        if type(row.batch_size) is not int:
            raise ValueError("calibration result batch_size must be an integer")
        if type(row.verifier_k) is not int:
            raise ValueError("calibration result verifier_k must be an integer")
        if not 0 <= row.verifier_k <= capability:
            name = "DFlash" if drafter == "dflash" else "DSpark"
            raise ValueError(f"{name} supports at most K{capability}")
        if row.verifier_k not in contract.calibration_k_values:
            if drafter == "dspark" and row.verifier_k == 8:
                raise ValueError(
                    "DSpark supports K8, but the current calibration grid through K7 "
                    "does not include it"
                )
            raise ValueError(
                "calibration result verifier_k is outside the required grid"
            )
        if type(row.repetition) is not int or row.repetition <= 0:
            raise ValueError("calibration result repetition must be a positive integer")
        if type(row.output_tokens) is not int or row.output_tokens <= 0:
            raise ValueError(
                "calibration result output_tokens must be a positive integer"
            )
        if (
            not isinstance(row.elapsed_seconds, (int, float))
            or isinstance(row.elapsed_seconds, bool)
            or not math.isfinite(row.elapsed_seconds)
            or row.elapsed_seconds <= 0
        ):
            raise ValueError(
                "calibration result elapsed_seconds must be positive and finite"
            )

        identity = (row.repetition, row.batch_size, row.verifier_k)
        if identity in identities:
            raise ValueError("duplicate calibration result for repetition/BS/K cell")
        identities.add(identity)

    required_cells = {
        (batch_size, verifier_k)
        for batch_size in contract.calibration_batch_sizes
        for verifier_k in contract.calibration_k_values
    }
    for repetition in sorted({row.repetition for row in candidates}):
        actual_cells = {
            (row.batch_size, row.verifier_k)
            for row in candidates
            if row.repetition == repetition
        }
        if actual_cells != required_cells:
            raise ValueError(
                f"repetition {repetition} does not contain the exact required BS/K grid"
            )
    return candidates


def _optimal_monotone_path(
    *,
    batch_sizes: list[int],
    k_values: list[int],
    cell_medians: dict[tuple[int, int], float],
) -> tuple[int, ...]:
    """Maximize summed throughput, preferring smaller-K paths on score ties."""
    states = {
        verifier_k: (
            cell_medians[(batch_sizes[0], verifier_k)],
            (verifier_k,),
        )
        for verifier_k in k_values
    }
    for batch_size in batch_sizes[1:]:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for verifier_k in k_values:
            candidates = [
                (
                    score + cell_medians[(batch_size, verifier_k)],
                    path + (verifier_k,),
                )
                for previous_k, (score, path) in states.items()
                if verifier_k <= previous_k
            ]
            best_score = max(score for score, _ in candidates)
            best_path = min(
                path for score, path in candidates if score == best_score
            )
            next_states[verifier_k] = (best_score, best_path)
        states = next_states

    best_score = max(score for score, _ in states.values())
    return min(path for score, path in states.values() if score == best_score)


def calibrate_drafter(
    rows: Iterable[CalibrationResultRow],
    *,
    drafter: Drafter,
) -> CalibrationSelection:
    """Select one drafter's best fixed K and monotone batch-size schedule."""
    candidates = _validated_rows(rows, drafter=drafter)
    cell_values: dict[tuple[int, int], list[float]] = {}
    for row in candidates:
        cell_values.setdefault((row.batch_size, row.verifier_k), []).append(
            row.output_tokens_per_second
        )
    cell_medians = {
        key: statistics.median(values) for key, values in cell_values.items()
    }
    batch_sizes = sorted({row.batch_size for row in candidates})
    k_values = sorted({row.verifier_k for row in candidates})

    selected_by_batch = dict(
        zip(
            batch_sizes,
            _optimal_monotone_path(
                batch_sizes=batch_sizes,
                k_values=k_values,
                cell_medians=cell_medians,
            ),
            strict=True,
        )
    )

    best_fixed_k = max(
        k_values,
        key=lambda verifier_k: (
            statistics.fmean(
                cell_medians[(batch_size, verifier_k)] for batch_size in batch_sizes
            ),
            -verifier_k,
        ),
    )

    schedule: list[list[int]] = []
    range_start = 1
    previous_batch_size = batch_sizes[0]
    range_k = selected_by_batch[previous_batch_size]
    for batch_size in batch_sizes[1:]:
        selected_k = selected_by_batch[batch_size]
        if selected_k != range_k:
            schedule.append([range_start, previous_batch_size, range_k])
            range_start = previous_batch_size + 1
            range_k = selected_k
        previous_batch_size = batch_size
    schedule.append([range_start, 128, range_k])
    return CalibrationSelection(drafter, best_fixed_k, schedule)
