from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pytest

from experiments.vllm_024_upgrade.summarize_qwen30_drafter_matrix import (
    _draft_revision_from_path,
    _target_revision_from_command,
    main,
)


BASE_TARGET = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
INSTRUCT_TARGET = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
THINKING_TARGET = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
BASE_DRAFT = "6afc5aa2477b923467fb9a8d906782b984a9a6ba"
INSTRUCT_DRAFT = "a7600ef6ca94c4e06cc1022879944be15949aee4"
THINKING_DRAFT = "a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf"


def _snapshot(repository: str, revision: str) -> str:
    return f"/models/hub/models--{repository}/snapshots/{revision}"


def _history(scale: float, *, healthy: bool = True) -> list[dict[str, float]]:
    return [
        {
            "_step": step,
            "timing/train/generation": 100.0 / scale,
            "timing/train/total_step_time": 200.0 / scale,
            "performance/generation_tokens_per_sec_per_gpu": 50.0 * scale,
            "performance/tokens_per_sec_per_gpu": 25.0 * scale,
            "train/vllm/spec_num_drafts": 100.0,
            "train/vllm/spec_num_draft_tokens": 300.0,
            "train/vllm/spec_num_accepted_tokens": 150.0,
            "train/reward": 0.4 if healthy else 0.2,
            "train/mean_gen_tokens_per_sample": 1024.0,
            "train/gen_kl_error": 0.01,
        }
        for step in range(1, 21)
    ]


class _FakeRun:
    def __init__(self, run_id: str, scale: float, variant: str) -> None:
        self.url = f"https://wandb.example/runs/{run_id}"
        self._scale = scale
        self._variant = variant

    def scan_history(self, *, keys: list[str]) -> Iterable[dict[str, float]]:
        if self._variant == "baseline":
            assert "train/vllm/spec_num_drafts" not in keys
        else:
            assert "train/vllm/spec_num_drafts" in keys
        return ({key: row[key] for key in keys} for row in _history(self._scale))


class _FakeApi:
    def __init__(self, runs: dict[str, tuple[float, str]]) -> None:
        self._runs = runs

    def run(self, path: str) -> _FakeRun:
        assert path.startswith("nvidia/nemorl-vllm024-q30-drafter-lyris/")
        run_id = path.rsplit("/", maxsplit=1)[-1]
        scale, variant = self._runs[run_id]
        return _FakeRun(run_id, scale, variant)


def _manifest_row(
    *,
    target_revision: str,
    draft_revision: str | None,
    variant: str,
    run_id: str,
    baseline_records_drafter: bool = False,
) -> dict[str, str]:
    target = _snapshot("Qwen--Qwen3-30B-A3B", target_revision)
    draft = (
        _snapshot("RedHatAI--Qwen3-30B-A3B-speculator.eagle3", draft_revision)
        if draft_revision is not None
        else ""
    )
    if variant == "baseline" and baseline_records_drafter:
        draft = _snapshot(
            "RedHatAI--Qwen3-30B-A3B-speculator.eagle3", BASE_DRAFT
        )
    return {
        "model": "qwen30ba3b",
        "variant": variant,
        "job_id": f"job-{run_id}",
        "wandb_run_id": run_id,
        "wandb_url": f"https://submitted.example/runs/{run_id}",
        "draft_model": draft,
        "rejection_sample_method": "standard" if draft else "",
        "draft_sample_method": "probabilistic" if draft else "",
        "command": (
            "env WANDB_RUN_ID=test python examples/run_grpo.py "
            f"policy.model_name={target} policy.tokenizer.name={target}"
        ),
    }


def _write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=list(materialized[0]),
        )
        writer.writeheader()
        writer.writerows(materialized)


def _matrix(tmp_path: Path) -> tuple[list[Path], dict[str, tuple[float, str]]]:
    definitions = {
        "base__base": (BASE_TARGET, BASE_DRAFT, True, 1.2, 1.5),
        "base__instruct2507": (BASE_TARGET, INSTRUCT_DRAFT, False, 1.1, 1.3),
        "instruct2507__instruct2507": (
            INSTRUCT_TARGET,
            INSTRUCT_DRAFT,
            True,
            1.25,
            1.4,
        ),
        "thinking2507__thinking2507": (
            THINKING_TARGET,
            THINKING_DRAFT,
            True,
            1.15,
            1.35,
        ),
    }
    manifests: list[Path] = []
    runs: dict[str, tuple[float, str]] = {}
    for pair_id, (
        target,
        draft,
        has_baseline,
        fixed_scale,
        dynamic_scale,
    ) in definitions.items():
        rows: list[dict[str, str]] = []
        if has_baseline:
            run_id = f"{pair_id}-baseline"
            rows.append(
                _manifest_row(
                    target_revision=target,
                    draft_revision=None,
                    variant="baseline",
                    run_id=run_id,
                )
            )
            runs[run_id] = (1.0, "baseline")
        for variant, scale in (("eagle3_k5", fixed_scale), ("dynamic", dynamic_scale)):
            run_id = f"{pair_id}-{variant}"
            rows.append(
                _manifest_row(
                    target_revision=target,
                    draft_revision=draft,
                    variant=variant,
                    run_id=run_id,
                )
            )
            runs[run_id] = (scale, variant)
        manifest = tmp_path / "runs" / pair_id / "submissions.tsv"
        _write_manifest(manifest, reversed(rows))
        manifests.append(manifest)
    return manifests, runs


def test_extracts_snapshot_revisions_from_serialized_provenance() -> None:
    target = _snapshot("Qwen--Qwen3-30B-A3B", BASE_TARGET)
    command = f"env A=1 python run.py policy.model_name={target} x=2"

    assert _target_revision_from_command(command) == BASE_TARGET
    assert (
        _draft_revision_from_path(_snapshot("RedHatAI--drafter", BASE_DRAFT))
        == BASE_DRAFT
    )


def test_target_revision_requires_policy_model_name_override() -> None:
    with pytest.raises(ValueError, match="policy.model_name"):
        _target_revision_from_command("python examples/run_grpo.py")


def test_baseline_ignores_provenance_only_drafter_path(tmp_path: Path) -> None:
    manifest = tmp_path / "base__base" / "submissions.tsv"
    run_id = "base-baseline"
    _write_manifest(
        manifest,
        [
            _manifest_row(
                target_revision=BASE_TARGET,
                draft_revision=None,
                variant="baseline",
                run_id=run_id,
                baseline_records_drafter=True,
            )
        ],
    )
    output_dir = tmp_path / "summary"

    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_FakeApi({run_id: (1.0, "baseline")}),
        )
        == 0
    )
    row = json.loads((output_dir / "summary.json").read_text())["rows"][0]
    assert row["draft_revision"] is None


def test_main_collects_multiple_manifests_with_revision_matched_comparisons(
    tmp_path: Path,
) -> None:
    manifests, runs = _matrix(tmp_path)
    output_dir = tmp_path / "summary"
    argv = [item for path in reversed(manifests) for item in ("--manifest", str(path))]

    exit_code = main(
        [*argv, "--output-dir", str(output_dir)],
        api=_FakeApi(runs),
    )

    assert exit_code == 0
    report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["metadata"]["aliases"] == [
        {
            "alias_pair_id": "base__thinking2507",
            "canonical_pair_id": "base__base",
            "reason": "identical_model_and_config_blobs",
        }
    ]
    rows = report["rows"]
    assert len(rows) == 11
    assert not any(row["pair_id"] == "base__thinking2507" for row in rows)

    dynamic = next(
        row
        for row in rows
        if row["pair_id"] == "base__instruct2507" and row["variant"] == "dynamic"
    )
    assert dynamic["target_revision"] == BASE_TARGET
    assert dynamic["draft_revision"] == INSTRUCT_DRAFT
    assert dynamic["generation_time_speedup_vs_baseline"] == pytest.approx(1.3)
    assert dynamic["e2e_throughput_speedup_vs_baseline"] == pytest.approx(1.3)
    assert dynamic["generation_throughput_speedup_vs_fixed"] == pytest.approx(1.3 / 1.1)
    assert dynamic["acceptance_rate"] == 0.5
    assert dynamic["mean_acceptance_length"] == 2.5
    assert dynamic["reward_health_passed"] is True
    assert dynamic["response_length_health_passed"] is True
    assert dynamic["kl_health_passed"] is True
    assert dynamic["health_gate_passed"] is True
    assert dynamic["job_id"] == "job-base__instruct2507-dynamic"
    assert dynamic["wandb_url"] == (
        "https://submitted.example/runs/base__instruct2507-dynamic"
    )
    assert dynamic["source_manifest"] == str(
        (tmp_path / "runs" / "base__instruct2507" / "submissions.tsv").resolve()
    )

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 11
    assert "generation_time_s" in csv_rows[0]
    assert "e2e_step_time_speedup_vs_fixed" in csv_rows[0]
    assert "reward" in csv_rows[0]


def test_outputs_are_byte_deterministic_across_manifest_argument_order(
    tmp_path: Path,
) -> None:
    manifests, runs = _matrix(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert (
        main(
            [
                *(item for path in manifests for item in ("--manifest", str(path))),
                "--output-dir",
                str(first),
            ],
            api=_FakeApi(runs),
        )
        == 0
    )
    assert (
        main(
            [
                *(
                    item
                    for path in reversed(manifests)
                    for item in ("--manifest", str(path))
                ),
                "--output-dir",
                str(second),
            ],
            api=_FakeApi(runs),
        )
        == 0
    )

    assert (first / "summary.json").read_bytes() == (
        second / "summary.json"
    ).read_bytes()
    assert (first / "summary.csv").read_bytes() == (second / "summary.csv").read_bytes()
