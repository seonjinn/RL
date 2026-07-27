# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from statistics import mean, median
from typing import Any


class TrainPhaseTimer:
    """Opt-in wall timer for Megatron worker phase diagnostics."""

    def __init__(
        self,
        *,
        enabled: bool,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None],
    ) -> None:
        self.enabled = enabled
        self._clock = clock
        self._synchronize = synchronize
        self.metrics: dict[str, float] = {}
        self._started_at: dict[str, float] = {}

    @classmethod
    def from_env(
        cls,
        *,
        synchronize: Callable[[], None],
        clock: Callable[[], float] = time.perf_counter,
    ) -> "TrainPhaseTimer":
        value = os.environ.get("NRL_MEGATRON_TRAIN_BREAKDOWN", "0")
        if value not in {"0", "1"}:
            raise ValueError("NRL_MEGATRON_TRAIN_BREAKDOWN must be 0 or 1")
        return cls(
            enabled=value == "1",
            clock=clock,
            synchronize=synchronize,
        )

    def start(self, label: str, *, synchronize_cuda: bool = False) -> None:
        if not self.enabled:
            return
        if label in self._started_at:
            raise ValueError(f"Train phase '{label}' is already running")
        if synchronize_cuda:
            self._synchronize()
        self._started_at[label] = self._clock()

    def stop(self, label: str, *, synchronize_cuda: bool = False) -> None:
        if not self.enabled:
            return
        if label not in self._started_at:
            raise ValueError(f"Train phase '{label}' is not running")
        if synchronize_cuda:
            self._synchronize()
        elapsed = self._clock() - self._started_at.pop(label)
        self.metrics[label] = self.metrics.get(label, 0.0) + elapsed

    @contextmanager
    def time(
        self, label: str, *, synchronize_cuda: bool = False
    ) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return

        self.start(label, synchronize_cuda=synchronize_cuda)
        try:
            yield
        finally:
            self.stop(label, synchronize_cuda=synchronize_cuda)


def aggregate_train_phase_timings(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Summarize phase timings across all worker ranks."""
    if not results or "train_phase_timings" not in results[0]:
        return {}

    expected_keys = set(results[0]["train_phase_timings"])
    for result in results:
        if set(result.get("train_phase_timings", {})) != expected_keys:
            raise ValueError("Every worker must report the same phase keys")

    critical_phase = (
        "worker_total" if "worker_total" in expected_keys else min(expected_keys)
    )
    critical_values = [
        float(result["train_phase_timings"][critical_phase]) for result in results
    ]
    critical_index = max(
        range(len(critical_values)),
        key=critical_values.__getitem__,
    )

    aggregated: dict[str, dict[str, float | int]] = {}
    for key in sorted(expected_keys):
        values = [float(result["train_phase_timings"][key]) for result in results]
        max_index = max(range(len(values)), key=values.__getitem__)
        aggregated[key] = {
            "min": min(values),
            "mean": mean(values),
            "median": median(values),
            "max": max(values),
            "max_rank": int(results[max_index].get("rank", max_index)),
            "critical_rank_value": values[critical_index],
        }
    return aggregated


def flatten_train_phase_timings(
    timings: dict[str, dict[str, float | int]],
) -> dict[str, float | int]:
    """Flatten rank-distribution timings for console and metric loggers."""
    return {
        f"worker_train/{phase}_{statistic}": value
        for phase, statistics in sorted(timings.items())
        for statistic, value in statistics.items()
        if statistic in {"min", "mean", "median", "max", "critical_rank_value"}
    }


def flatten_train_phase_metadata(
    timings: dict[str, dict[str, float | int]],
) -> dict[str, int]:
    """Flatten rank identifiers separately from duration metrics."""
    metadata = {
        f"worker_train/{phase}_max_rank": int(statistics["max_rank"])
        for phase, statistics in sorted(timings.items())
    }
    if "worker_total" in timings:
        metadata["worker_train/critical_rank"] = int(
            timings["worker_total"]["max_rank"]
        )
    return metadata
