from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import pytest

from experiments.vllm_024_upgrade.summarize_tail_gated_specdec import (
    REQUIRED_MANIFEST_FIELDS,
)
from experiments.vllm_024_upgrade.validate_mini_sync_grpo_tail_gate import main


class _FakeRun:
    def __init__(self, history: list[dict[str, object]], url: str) -> None:
        self._history = history
        self.url = url

    def scan_history(self, *, keys: list[str]) -> Iterable[Mapping[str, object]]:
        del keys
        return self._history


class _FakeApi:
    def __init__(self, histories: dict[str, list[dict[str, object]]]) -> None:
        self._histories = histories
        self.calls: list[str] = []

    def run(self, path: str) -> _FakeRun:
        self.calls.append(path)
        run_id = path.rsplit("/", maxsplit=1)[-1]
        return _FakeRun(self._histories[run_id], f"https://wandb.example/{run_id}")


def _metadata(variant: str) -> dict[str, str]:
    gated = variant == "fastrl_threshold_v2_k5"
    values = {
        "timestamp": "2026-07-10T12:00:00Z",
        "model": "qwen32b",
        "variant": variant,
        "gate_mode": "threshold" if gated else "off",
        "k": "0" if variant == "baseline_v2" else "5",
        "threshold": "32" if gated else "",
        "consecutive_checks": "10" if gated else "",
        "roofline_config_sha256": "",
        "cluster": "pre-tyche",
        "runtime": "nemo-rl",
        "runtime_version": "nightly-20260705",
        "runtime_commit": "abc123",
        "vllm_version": "0.24.0",
        "vllm_commit": "ee0da84a",
        "target_tp": "4",
        "draft_tp": "1",
        "dp": "4",
        "ep": "1",
        "temperature": "1.0",
        "top_p": "1.0",
        "max_osl": "1024",
        "max_model_len": "1056",
        "max_sequence_length": "1024",
        "num_prompts": "16",
        "num_generations": "4",
        "train_gbs": "64",
        "max_num_batched_tokens": "4096",
        "max_num_seqs": "64",
        "recipe": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml",
        "container": "/containers/nemo.sqsh",
        "container_sha256": "deadbeef",
        "runner": "v2",
        "graph_mode": "FULL_AND_PIECEWISE",
        "sampling": "standard",
        "job_id": f"job-{variant}",
        "wandb_run_id": f"run-{variant}",
        "wandb_url": "",
    }
    assert set(REQUIRED_MANIFEST_FIELDS).issubset(values)
    return values


def _history(
    metadata: Mapping[str, str],
    *,
    activation_tick: float = 17.0,
    activation_batch: float = 16.0,
    enabled_ratio: float = 0.25,
    advance_only_ratio: float = 0.75,
    k0_steps: float = 75.0,
    k5_steps: float = 25.0,
    num_drafts: float = 100.0,
    num_accepted_tokens: float = 150.0,
    reward: float = 0.4,
    policy_time: float = 30.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in (1, 2):
        row = {
            "_step": step,
            "timing/train/total_step_time": 200.0,
            "timing/train/generation": 100.0,
            "performance/tokens_per_sec_per_gpu": 25.0,
            "performance/generation_tokens_per_sec_per_gpu": 50.0,
            "timing/train/policy_training": policy_time,
            "timing/train/policy_and_reference_logprobs": 20.0,
            "train/reward": reward,
            "train/mean_gen_tokens_per_sample": 512.0,
            "train/gen_kl_error": 0.01,
            "train/loss": 0.2,
            "train/vllm/cudagraph_target_graph_call_ratio": 1.0,
        }
        if metadata["variant"] != "baseline_v2":
            row.update(
                {
                    "train/vllm/spec_num_drafts": num_drafts,
                    "train/vllm/spec_num_draft_tokens": 300.0,
                    "train/vllm/spec_num_accepted_tokens": num_accepted_tokens,
                    "train/vllm/spec_acceptance_rate": 0.5,
                    "train/vllm/spec_acceptance_length": 2.5,
                    "train/vllm/cudagraph_draft_prefill_graph_call_ratio": 1.0,
                    "train/vllm/cudagraph_draft_decode_graph_call_ratio": 1.0,
                }
            )
        if metadata["gate_mode"] == "threshold":
            row.update(
                {
                    "train/vllm/tail_gate_decisions": 100.0,
                    "train/vllm/tail_gate_activations": 1.0,
                    "train/vllm/tail_gate_enabled_step_ratio": enabled_ratio,
                    "train/vllm/tail_gate_advance_only_step_ratio": advance_only_ratio,
                    "train/vllm/tail_gate_activation_tick": activation_tick,
                    "train/vllm/tail_gate_activation_batch": activation_batch,
                    "train/vllm/tail_gate_activation_seq_len": 512.0,
                    "train/vllm/tail_gate_predicted_speedup": 1.1,
                    "train/vllm/tail_gate_k_0_steps": k0_steps,
                    "train/vllm/tail_gate_k_5_steps": k5_steps,
                }
            )
        rows.append(row)
    return rows


def _cohort() -> list[dict[str, str]]:
    return [
        _metadata("baseline_v2"),
        _metadata("always_on_v2_k5"),
        _metadata("fastrl_threshold_v2_k5"),
    ]


def _write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(materialized[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(materialized)


def _run_validator(
    tmp_path: Path,
    rows: list[dict[str, str]],
    histories: dict[str, list[dict[str, object]]],
) -> tuple[int, Path]:
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    _write_manifest(manifest, rows)
    result = main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir)],
        api=_FakeApi(histories),
    )
    return result, output_dir


def test_mini_validator_exports_main() -> None:
    assert callable(main)


def test_mini_validator_accepts_completed_matched_threshold_smoke(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}

    result, output_dir = _run_validator(tmp_path, rows, histories)

    assert result == 0
    payload = json.loads((output_dir / "mini_summary.json").read_text())
    threshold = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert threshold["status"] == "final"
    assert threshold["mini_health_passed"] is True
    assert threshold["activation_tick"] == 17.0
    assert threshold["tail_gate_k0_steps"] == 75.0
    assert threshold["tail_gate_k5_steps"] == 25.0


def test_mini_validator_rejects_incomplete_matrix_before_wandb_query(
    tmp_path: Path,
) -> None:
    rows = _cohort()[:2]
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="mini manifest variants must be exactly"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("variant", "field", "invalid_value"),
    [
        ("baseline_v2", "gate_mode", "threshold"),
        ("baseline_v2", "k", "5"),
        ("baseline_v2", "threshold", "32"),
        ("baseline_v2", "consecutive_checks", "10"),
        ("always_on_v2_k5", "gate_mode", "threshold"),
        ("always_on_v2_k5", "k", "0"),
        ("always_on_v2_k5", "threshold", "32"),
        ("always_on_v2_k5", "consecutive_checks", "10"),
        ("fastrl_threshold_v2_k5", "gate_mode", "off"),
        ("fastrl_threshold_v2_k5", "k", "0"),
        ("fastrl_threshold_v2_k5", "threshold", "31"),
        ("fastrl_threshold_v2_k5", "consecutive_checks", "9"),
    ],
)
def test_mini_validator_rejects_invalid_variant_mapping_before_wandb_query(
    tmp_path: Path, variant: str, field: str, invalid_value: str
) -> None:
    rows = _cohort()
    target = next(row for row in rows if row["variant"] == variant)
    target[field] = invalid_value
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(
        ValueError,
        match=rf"invalid mini manifest field:{variant}:{field}",
    ):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


def test_mini_validator_uses_manifest_wandb_url_for_each_run(tmp_path: Path) -> None:
    rows = _cohort()
    for row in rows:
        row["wandb_url"] = (
            "https://wandb.ai/manifest-entity/manifest-project/runs/"
            f"{row['wandb_run_id']}"
        )
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    result = main(
        [
            "--manifest",
            str(manifest),
            "--entity",
            "wrong-entity",
            "--project",
            "wrong-project",
            "--output-dir",
            str(output_dir),
        ],
        api=api,
    )

    assert result == 0
    assert api.calls == [
        f"manifest-entity/manifest-project/{row['wandb_run_id']}" for row in rows
    ]
    payload = json.loads((output_dir / "mini_summary.json").read_text())
    urls_by_variant = {row["variant"]: row["wandb_url"] for row in payload}
    assert urls_by_variant == {row["variant"]: row["wandb_url"] for row in rows}


def test_mini_validator_fallback_matches_mini_launcher_project(tmp_path: Path) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    result = main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir)],
        api=api,
    )

    assert result == 0
    assert api.calls == [
        "nvidia/nemorl-vllm024-tail-gated-mini-sync-grpo-pre-tyche/"
        f"{row['wandb_run_id']}"
        for row in rows
    ]


def test_mini_validator_rejects_wandb_url_run_id_mismatch_before_query(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    rows[0]["wandb_url"] = "https://wandb.ai/nvidia/project/runs/different-run"
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    manifest = tmp_path / "submissions.tsv"
    output_dir = tmp_path / "output"
    api = _FakeApi(histories)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="wandb_url run ID mismatch:baseline_v2"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("updates", "failure"),
    [
        ({"activation_tick": 0.0}, "activation_tick"),
        ({"activation_batch": 33.0}, "activation_batch"),
        ({"enabled_ratio": 1.0}, "gate_enabled_ratio"),
        ({"advance_only_ratio": 0.0}, "gate_advance_only_ratio"),
        ({"k0_steps": 0.0}, "tail_gate_k0_steps"),
        ({"k5_steps": 0.0}, "tail_gate_k5_steps"),
        ({"num_drafts": 0.0}, "num_drafts"),
        ({"num_accepted_tokens": 0.0}, "num_accepted_tokens"),
        ({"reward": math.nan}, "reward"),
        ({"policy_time": 0.0}, "policy_training"),
    ],
)
def test_mini_validator_rejects_failed_threshold_health_gate(
    tmp_path: Path, updates: dict[str, float], failure: str
) -> None:
    rows = _cohort()
    histories = {
        row["wandb_run_id"]: _history(
            row,
            **(updates if row["variant"] == "fastrl_threshold_v2_k5" else {}),
        )
        for row in rows
    }

    result, output_dir = _run_validator(tmp_path, rows, histories)

    payload = json.loads((output_dir / "mini_summary.json").read_text())
    threshold = next(
        row for row in payload if row["variant"] == "fastrl_threshold_v2_k5"
    )
    assert result == 1
    assert threshold["status"] in {"partial", "health_failed"}
    assert failure in threshold["reason"]


def test_mini_validator_reuses_exact_collector_cohort_matching(tmp_path: Path) -> None:
    rows = _cohort()
    rows[1]["container_sha256"] = "different"
    histories = {row["wandb_run_id"]: _history(row) for row in rows}

    result, output_dir = _run_validator(tmp_path, rows, histories)

    payload = json.loads((output_dir / "mini_summary.json").read_text())
    assert result == 1
    assert all(row["status"] == "partial" for row in payload)
    reasons = {row["reason"] for row in payload}
    assert all(
        "comparison_failed:missing matched always-on" in reason for reason in reasons
    )


def test_activation_scatter_is_deterministic_and_identifies_the_event(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    histories = {row["wandb_run_id"]: _history(row) for row in rows}
    first_result, first_output = _run_validator(tmp_path / "first", rows, histories)
    second_result, second_output = _run_validator(
        tmp_path / "second", list(reversed(rows)), histories
    )

    assert first_result == second_result == 0
    first = (first_output / "tail_gate_activation_events.html").read_bytes()
    second = (second_output / "tail_gate_activation_events.html").read_bytes()
    assert first == second
    report = first.decode()
    assert "Scheduler tick" in report
    assert "Inflight batch" in report
    assert "threshold=32" in report
    assert "OFF-to-ON" in report
    assert "tick=17" in report
    assert "batch=16" in report
    assert "stable speedup" not in report
    assert "two-step smoke makes no speedup claim" in report
