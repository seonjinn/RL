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

import argparse
import csv
import json
import math
import os
import shlex
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from experiments.vllm_024_upgrade.summarize_eagle3_dynamicsd import (
    RunSummary,
    WandbApi,
    _create_wandb_api,
    _history_keys,
    summarize_history,
)


REPORT_METADATA = {
    "matrix": "qwen30_drafter_distribution",
    "aggregation_steps": list(range(2, 21)),
    "aliases": [
        {
            "alias_pair_id": "base__thinking2507",
            "canonical_pair_id": "base__base",
            "reason": "identical_model_and_config_blobs",
        }
    ],
}
SUPPORTED_VARIANTS = frozenset({"baseline", "eagle3_k5", "dynamic"})
SUMMARY_FIELDS = (
    "model",
    "variant",
    "rejection_sample_method",
    "draft_sample_method",
    "complete",
    "reason",
    "measured_steps",
    "generation_time_s",
    "e2e_step_time_s",
    "generation_throughput",
    "e2e_throughput",
    "acceptance_rate",
    "mean_acceptance_length",
    "reward",
    "mean_response_length",
    "approx_kl",
)
COMPARISON_FIELDS = (
    "generation_time_speedup_vs_baseline",
    "e2e_step_time_speedup_vs_baseline",
    "generation_throughput_speedup_vs_baseline",
    "e2e_throughput_speedup_vs_baseline",
    "generation_time_speedup_vs_fixed",
    "e2e_step_time_speedup_vs_fixed",
    "generation_throughput_speedup_vs_fixed",
    "e2e_throughput_speedup_vs_fixed",
    "reward_health_passed",
    "response_length_health_passed",
    "kl_health_passed",
    "health_gate_passed",
)
CSV_FIELDS = (
    "pair_id",
    "target_revision",
    "draft_revision",
    "source_manifest",
    "job_id",
    "wandb_run_id",
    "wandb_url",
    *SUMMARY_FIELDS,
    *COMPARISON_FIELDS,
)


@dataclass(frozen=True)
class ManifestRun:
    pair_id: str
    target_revision: str
    draft_revision: str | None
    source_manifest: str
    job_id: str
    wandb_run_id: str
    wandb_url: str
    model: str
    variant: str
    rejection_sample_method: str
    draft_sample_method: str


@dataclass(frozen=True)
class CollectedRun:
    manifest: ManifestRun
    summary: RunSummary


def _revision_from_snapshot_path(value: str, *, field: str) -> str:
    parts = PurePosixPath(value).parts
    snapshot_indexes = [
        index for index, part in enumerate(parts) if part == "snapshots"
    ]
    if len(snapshot_indexes) != 1:
        raise ValueError(f"{field} must contain one snapshots/<revision> component")
    revision_index = snapshot_indexes[0] + 1
    if revision_index >= len(parts) or revision_index != len(parts) - 1:
        raise ValueError(f"{field} must end in snapshots/<revision>")
    revision = parts[revision_index]
    if not revision:
        raise ValueError(f"{field} has an empty snapshot revision")
    return revision


def _target_revision_from_command(command: str) -> str:
    try:
        arguments = shlex.split(command)
    except ValueError as error:
        raise ValueError(f"invalid serialized command: {error}") from error
    prefix = "policy.model_name="
    model_paths = {
        argument.removeprefix(prefix)
        for argument in arguments
        if argument.startswith(prefix)
    }
    if len(model_paths) != 1:
        raise ValueError(
            "serialized command must contain one policy.model_name override"
        )
    return _revision_from_snapshot_path(model_paths.pop(), field="policy.model_name")


def _draft_revision_from_path(draft_model: str) -> str:
    return _revision_from_snapshot_path(draft_model, field="draft_model")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _manifest_runs(paths: Iterable[Path]) -> list[ManifestRun]:
    resolved_paths = [path.resolve() for path in paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("duplicate --manifest path")

    runs: list[ManifestRun] = []
    for path in sorted(resolved_paths):
        manifest_rows = _read_manifest(path)
        if not manifest_rows:
            raise ValueError(f"empty manifest: {path}")
        pair_id = path.parent.name
        for row in manifest_rows:
            variant = row.get("variant", "")
            if variant not in SUPPORTED_VARIANTS:
                raise ValueError(f"unsupported variant {variant!r} in {path}")
            command = row.get("command", "")
            target_revision = _target_revision_from_command(command)
            draft_model = row.get("draft_model", "")
            if variant == "baseline":
                draft_revision = None
            else:
                if not draft_model:
                    raise ValueError(f"{variant} is missing draft_model in {path}")
                draft_revision = _draft_revision_from_path(draft_model)

            model = row.get("model", "")
            wandb_run_id = row.get("wandb_run_id", "")
            if not model or not wandb_run_id:
                raise ValueError(f"missing model or wandb_run_id in {path}")
            runs.append(
                ManifestRun(
                    pair_id=pair_id,
                    target_revision=target_revision,
                    draft_revision=draft_revision,
                    source_manifest=str(path),
                    job_id=row.get("job_id", ""),
                    wandb_run_id=wandb_run_id,
                    wandb_url=row.get("wandb_url", ""),
                    model=model,
                    variant=variant,
                    rejection_sample_method=(
                        row.get("rejection_sample_method", "") or "not_applicable"
                    ),
                    draft_sample_method=(
                        row.get("draft_sample_method", "") or "not_applicable"
                    ),
                )
            )

    identities = [
        (
            run.target_revision,
            run.draft_revision,
            run.variant,
            run.rejection_sample_method,
            run.draft_sample_method,
        )
        for run in runs
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate target/draft/variant identity")
    return sorted(runs, key=_manifest_sort_key)


def _manifest_sort_key(run: ManifestRun) -> tuple[str, str, str, str, str]:
    return (
        run.target_revision,
        run.draft_revision or "",
        run.variant,
        run.pair_id,
        run.wandb_run_id,
    )


def _empty_summary(run: ManifestRun, reason: str) -> RunSummary:
    return RunSummary(
        model=run.model,
        variant=run.variant,
        rejection_sample_method=run.rejection_sample_method,
        draft_sample_method=run.draft_sample_method,
        complete=False,
        reason=reason,
        measured_steps=[],
        generation_time_s=None,
        e2e_step_time_s=None,
        generation_throughput=None,
        e2e_throughput=None,
        acceptance_rate=None,
        mean_acceptance_length=None,
        reward=None,
        mean_response_length=None,
        approx_kl=None,
    )


def _collect_runs(
    manifest_runs: Iterable[ManifestRun],
    *,
    entity: str,
    project: str,
    api: WandbApi,
) -> list[CollectedRun]:
    collected: list[CollectedRun] = []
    for manifest_run in manifest_runs:
        try:
            wandb_run = api.run(f"{entity}/{project}/{manifest_run.wandb_run_id}")
            summary = summarize_history(
                manifest_run.model,
                manifest_run.variant,
                wandb_run.scan_history(keys=_history_keys(manifest_run.variant)),
                rejection_sample_method=manifest_run.rejection_sample_method,
                draft_sample_method=manifest_run.draft_sample_method,
            )
            if not manifest_run.wandb_url:
                manifest_run = ManifestRun(
                    **{**asdict(manifest_run), "wandb_url": wandb_run.url}
                )
        except Exception as error:  # W&B failures remain visible in the report.
            summary = _empty_summary(
                manifest_run,
                f"wandb_fetch_failed:{type(error).__name__}",
            )
        collected.append(CollectedRun(manifest=manifest_run, summary=summary))
    return collected


def _speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _health_metric(candidate: float | None, baseline: float | None) -> bool:
    if candidate is None or baseline is None:
        return False
    if baseline == 0.0:
        return math.isclose(candidate, 0.0, abs_tol=1e-8)
    return abs(candidate - baseline) / abs(baseline) <= 0.10


def _index_unique(
    runs: Iterable[CollectedRun],
    *,
    variant: str,
    include_draft: bool,
) -> dict[tuple[str, ...], CollectedRun]:
    index: dict[tuple[str, ...], CollectedRun] = {}
    for run in runs:
        if run.manifest.variant != variant:
            continue
        key = (run.manifest.target_revision,)
        if include_draft:
            key += (run.manifest.draft_revision or "",)
        if key in index:
            raise ValueError(f"duplicate {variant} comparison identity: {key}")
        index[key] = run
    return index


def _comparison_rows(runs: Iterable[CollectedRun]) -> list[dict[str, object]]:
    materialized = list(runs)
    baselines = _index_unique(
        materialized,
        variant="baseline",
        include_draft=False,
    )
    fixed_runs = _index_unique(
        materialized,
        variant="eagle3_k5",
        include_draft=True,
    )
    rows: list[dict[str, object]] = []
    for run in materialized:
        manifest = run.manifest
        summary = run.summary
        baseline_run = baselines.get((manifest.target_revision,))
        fixed_run = fixed_runs.get(
            (manifest.target_revision, manifest.draft_revision or "")
        )
        comparison_error = ""
        if baseline_run is None:
            comparison_error = "missing_baseline_for_target_revision"
        elif not baseline_run.summary.complete:
            comparison_error = "incomplete_baseline_for_target_revision"
        elif manifest.variant == "dynamic" and fixed_run is None:
            comparison_error = "missing_fixed_for_target_and_draft_revision"
        elif (
            manifest.variant == "dynamic"
            and fixed_run is not None
            and not fixed_run.summary.complete
        ):
            comparison_error = "incomplete_fixed_for_target_and_draft_revision"

        baseline = baseline_run.summary if baseline_run is not None else None
        fixed = fixed_run.summary if fixed_run is not None else None
        reward_health = _health_metric(
            summary.reward,
            baseline.reward if baseline is not None else None,
        )
        response_health = _health_metric(
            summary.mean_response_length,
            baseline.mean_response_length if baseline is not None else None,
        )
        kl_health = _health_metric(
            summary.approx_kl,
            baseline.approx_kl if baseline is not None else None,
        )
        row: dict[str, object] = {
            **asdict(manifest),
            **asdict(summary),
            "generation_time_speedup_vs_baseline": _speedup(
                baseline.generation_time_s if baseline is not None else None,
                summary.generation_time_s,
            ),
            "e2e_step_time_speedup_vs_baseline": _speedup(
                baseline.e2e_step_time_s if baseline is not None else None,
                summary.e2e_step_time_s,
            ),
            "generation_throughput_speedup_vs_baseline": _speedup(
                summary.generation_throughput,
                baseline.generation_throughput if baseline is not None else None,
            ),
            "e2e_throughput_speedup_vs_baseline": _speedup(
                summary.e2e_throughput,
                baseline.e2e_throughput if baseline is not None else None,
            ),
            "generation_time_speedup_vs_fixed": (
                _speedup(fixed.generation_time_s, summary.generation_time_s)
                if manifest.variant == "dynamic" and fixed is not None
                else None
            ),
            "e2e_step_time_speedup_vs_fixed": (
                _speedup(fixed.e2e_step_time_s, summary.e2e_step_time_s)
                if manifest.variant == "dynamic" and fixed is not None
                else None
            ),
            "generation_throughput_speedup_vs_fixed": (
                _speedup(summary.generation_throughput, fixed.generation_throughput)
                if manifest.variant == "dynamic" and fixed is not None
                else None
            ),
            "e2e_throughput_speedup_vs_fixed": (
                _speedup(summary.e2e_throughput, fixed.e2e_throughput)
                if manifest.variant == "dynamic" and fixed is not None
                else None
            ),
            "reward_health_passed": reward_health,
            "response_length_health_passed": response_health,
            "kl_health_passed": kl_health,
            "health_gate_passed": reward_health and response_health and kl_health,
        }
        if comparison_error:
            row["complete"] = False
            row["reason"] = f"comparison_failed:{comparison_error}"
        rows.append(row)
    return rows


def _write_json_atomic(path: Path, report: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        fieldnames: list[str] = list(CSV_FIELDS)
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the Qwen3-30B drafter distribution matrix."
    )
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--entity", default="nvidia")
    parser.add_argument("--project", default="nemorl-vllm024-q30-drafter-lyris")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, api: WandbApi | None = None) -> int:
    args = _parse_args(argv)
    manifest_runs = _manifest_runs(args.manifest)
    client = api if api is not None else _create_wandb_api()
    collected = _collect_runs(
        manifest_runs,
        entity=args.entity,
        project=args.project,
        api=client,
    )
    rows = _comparison_rows(collected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        args.output_dir / "summary.json",
        {"metadata": REPORT_METADATA, "rows": rows},
    )
    _write_csv_atomic(args.output_dir / "summary.csv", rows)
    return int(
        any(not bool(row["complete"]) for row in rows)
        or any(
            row["variant"] != "baseline" and row["health_gate_passed"] is False
            for row in rows
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
