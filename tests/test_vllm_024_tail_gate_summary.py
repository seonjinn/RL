from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.vllm_024_upgrade.summarize_tail_gated_specdec import (
    METRIC_KEYS,
    REQUIRED_ROW_FIELDS,
    build_comparison_rows,
    main,
    summarize_history,
)


def _history(
    *,
    scale: float = 1.0,
    graph_ratio: float = 1.0,
    reward: float = 0.4,
) -> list[dict[str, float]]:
    return [
        {
            "_step": step,
            "timing/train/total_step_time": 200.0 / scale,
            "timing/train/generation": 100.0 / scale,
            "performance/tokens_per_sec_per_gpu": 25.0 * scale,
            "performance/generation_tokens_per_sec_per_gpu": 50.0 * scale,
            "timing/train/policy": 30.0,
            "timing/train/logprob": 20.0,
            "train/vllm/spec_acceptance_rate": 0.5,
            "train/vllm/spec_acceptance_length": 2.5,
            "train/vllm/spec_gate_enabled_ratio": 0.25,
            "train/vllm/spec_gate_activation_batch": 16.0,
            "train/vllm/spec_gate_activation_seq_len": 2048.0,
            "train/vllm/spec_predicted_speedup": 1.1,
            "train/vllm/target_graph_ratio": graph_ratio,
            "train/vllm/draft_prefill_graph_ratio": graph_ratio,
            "train/vllm/draft_decode_graph_ratio": graph_ratio,
            "train/reward": reward,
            "train/mean_gen_tokens_per_sample": 1024.0,
            "train/gen_kl_error": 0.01,
            "train/policy_loss": 0.2,
        }
        for step in range(1, 21)
    ]


def _metadata(
    *,
    model: str = "qwen32b",
    runner: str = "v2",
    graph_mode: str = "FULL_AND_PIECEWISE",
    variant: str = "baseline_v2",
) -> dict[str, str]:
    return {
        "model": model,
        "runner": runner,
        "variant": variant,
        "gate_mode": "off" if variant.startswith("baseline") else "threshold",
        "k": "0" if variant.startswith("baseline") else "5",
        "graph_mode": graph_mode,
        "recipe": "recipe.yaml",
        "commit": "abc123",
        "container": "/containers/nemo.sqsh",
        "container_sha256": "deadbeef",
        "job_id": "12345",
        "wandb_run_id": f"run-{variant}",
        "wandb_url": f"https://wandb.example/{variant}",
        "source": "submissions-20260710.tsv",
    }


def _summary(
    metadata: dict[str, str],
    *,
    scale: float = 1.0,
    graph_ratio: float = 1.0,
    reward: float = 0.4,
):
    return summarize_history(
        metadata,
        _history(scale=scale, graph_ratio=graph_ratio, reward=reward),
    )


def test_summarize_history_averages_only_steps_2_through_20_with_full_contract() -> (
    None
):
    summary = _summary(_metadata())

    assert summary.status == "final"
    assert summary.steps == list(range(2, 21))
    assert summary.e2e_time == 200.0
    assert summary.generation_time == 100.0
    assert summary.gate_enabled_ratio == 0.25
    assert summary.target_graph_ratio == 1.0
    assert summary.wandb_url == "https://wandb.example/baseline_v2"
    assert set(METRIC_KEYS).issubset(summary.to_dict())
    row = build_comparison_rows([summary])[0]
    assert set(REQUIRED_ROW_FIELDS).issubset(row.to_dict())


def test_incomplete_history_is_labeled_partial_and_preserves_provenance() -> None:
    metadata = _metadata()
    summary = summarize_history(metadata, _history()[:-1])

    assert summary.status == "partial"
    assert summary.reason == "missing_steps:20"
    assert summary.job_id == "12345"
    assert summary.wandb_url == "https://wandb.example/baseline_v2"


def test_comparisons_use_only_exact_runner_graph_and_config_baselines() -> None:
    baseline = _summary(_metadata())
    candidate = _summary(_metadata(variant="fastrl_threshold_v2_k5"), scale=1.25)
    rows = build_comparison_rows([baseline, candidate])

    by_variant = {row.variant: row for row in rows}
    assert by_variant["fastrl_threshold_v2_k5"].e2e_time_speedup_vs_baseline == 1.25
    assert by_variant["fastrl_threshold_v2_k5"].e2e_tps_gpu_speedup_vs_baseline == 1.25
    assert by_variant["fastrl_threshold_v2_k5"].status == "final"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runner", "v1"),
        ("graph_mode", "PIECEWISE"),
        ("commit", "different-commit"),
        ("container_sha256", "different-container"),
        ("recipe", "different-recipe.yaml"),
    ],
)
def test_comparisons_reject_cross_cohort_or_cross_config_speedups(
    field: str, value: str
) -> None:
    baseline = _summary(_metadata())
    candidate_metadata = _metadata(variant="fastrl_threshold_v2_k5")
    candidate_metadata[field] = value

    with pytest.raises(ValueError, match="missing matched baseline"):
        build_comparison_rows([baseline, _summary(candidate_metadata, scale=1.25)])


def test_health_and_cuda_graph_gates_are_reported() -> None:
    baseline = _summary(_metadata())
    unhealthy = _summary(
        _metadata(variant="fastrl_threshold_v2_k5"), graph_ratio=0.9, reward=0.0
    )

    row = next(
        row
        for row in build_comparison_rows([baseline, unhealthy])
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert row.reward_health_passed is False
    assert row.cuda_graph_health_passed is False
    assert row.health_gate_passed is False


class _FakeRun:
    def __init__(self, history: list[dict[str, float]], url: str) -> None:
        self._history = history
        self.url = url

    def scan_history(self, *, keys: list[str]):
        return iter({key: row[key] for key in keys} for row in self._history)


class _FakeApi:
    def run(self, path: str) -> _FakeRun:
        run_id = path.rsplit("/", maxsplit=1)[-1]
        if "baseline" in run_id:
            return _FakeRun(_history(), "https://wandb.example/from-api-baseline")
        return _FakeRun(
            _history(scale=1.25), "https://wandb.example/from-api-candidate"
        )


def test_main_writes_deterministic_validated_artifacts_and_runner_html(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "submissions.tsv"
    fieldnames = list(_metadata())
    rows = [
        _metadata(),
        _metadata(variant="fastrl_threshold_v2_k5"),
    ]
    rows[0]["wandb_url"] = ""
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reversed(rows))

    output_dir = tmp_path / "summary"
    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_FakeApi(),
        )
        == 0
    )

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert [row["variant"] for row in payload] == [
        "baseline_v2",
        "fastrl_threshold_v2_k5",
    ]
    assert payload[0]["wandb_url"] == "https://wandb.example/from-api-baseline"
    assert payload[1]["e2e_time_speedup_vs_baseline"] == 1.25
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        assert csv.DictReader(stream).fieldnames == list(REQUIRED_ROW_FIELDS)
    fragment = (output_dir / "tail_gated_specdec.html").read_text(encoding="utf-8")
    assert "Model Runner V2" in fragment
    assert "Model Runner V1" not in fragment
    assert "https://wandb.example/from-api-baseline" in fragment
    assert "Final finding:" in fragment


def test_main_refuses_to_overwrite_historical_output(tmp_path: Path) -> None:
    manifest = tmp_path / "submissions.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=list(_metadata()))
        writer.writeheader()
        writer.writerows([_metadata(), _metadata(variant="fastrl_threshold_v2_k5")])
    output_dir = tmp_path / "summary"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text("historical\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_FakeApi(),
        )


def test_main_retains_unmatched_run_as_a_visible_partial_row(tmp_path: Path) -> None:
    manifest = tmp_path / "submissions.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=list(_metadata()))
        writer.writeheader()
        writer.writerow(_metadata(variant="fastrl_threshold_v2_k5"))

    output_dir = tmp_path / "summary"
    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_FakeApi(),
        )
        == 1
    )

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload[0]["status"] == "partial"
    assert payload[0]["reason"].startswith("comparison_failed:missing matched baseline")
    assert 'class="partial"' in (output_dir / "tail_gated_specdec.html").read_text(
        encoding="utf-8"
    )
