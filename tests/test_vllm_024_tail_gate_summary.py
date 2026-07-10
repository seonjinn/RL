from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import pytest

from experiments.vllm_024_upgrade.summarize_tail_gated_specdec import (
    COHORT_FIELDS,
    METRIC_KEYS,
    REQUIRED_ROW_FIELDS,
    RunSummary,
    _history_keys,
    _validate_manifest_rows,
    build_comparison_rows,
    main,
    summarize_history,
)


EXPECTED_METRIC_KEYS = {
    "e2e_time": "timing/train/total_step_time",
    "generation_time": "timing/train/generation",
    "e2e_tps_gpu": "performance/tokens_per_sec_per_gpu",
    "generation_tps_gpu": "performance/generation_tokens_per_sec_per_gpu",
    "policy_time": "timing/train/policy_training",
    "logprob_time": "timing/train/policy_and_reference_logprobs",
    "reward": "train/reward",
    "response_length": "train/mean_gen_tokens_per_sample",
    "approx_kl": "train/gen_kl_error",
    "policy_loss": "train/loss",
    "num_drafts": "train/vllm/spec_num_drafts",
    "num_draft_tokens": "train/vllm/spec_num_draft_tokens",
    "num_accepted_tokens": "train/vllm/spec_num_accepted_tokens",
    "acceptance_rate": "train/vllm/spec_acceptance_rate",
    "mean_accept_len": "train/vllm/spec_acceptance_length",
    "gate_decisions": "train/vllm/tail_gate_decisions",
    "gate_activations": "train/vllm/tail_gate_activations",
    "gate_enabled_ratio": "train/vllm/tail_gate_enabled_step_ratio",
    "gate_advance_only_ratio": "train/vllm/tail_gate_advance_only_step_ratio",
    "activation_tick": "train/vllm/tail_gate_activation_tick",
    "activation_batch": "train/vllm/tail_gate_activation_batch",
    "activation_seq_len": "train/vllm/tail_gate_activation_seq_len",
    "predicted_speedup": "train/vllm/tail_gate_predicted_speedup",
    "activation_predicted_speedup": (
        "train/vllm/tail_gate_activation_predicted_speedup"
    ),
    "target_graph_ratio": "train/vllm/cudagraph_target_graph_call_ratio",
    "draft_graph_ratio": "train/vllm/cudagraph_draft_graph_call_ratio",
    "draft_prefill_graph_ratio": "train/vllm/cudagraph_draft_prefill_graph_call_ratio",
    "draft_decode_graph_ratio": "train/vllm/cudagraph_draft_decode_graph_call_ratio",
}


def _metadata(
    *,
    model: str = "qwen32b",
    runner: str = "v2",
    variant: str = "baseline_v2",
) -> dict[str, str]:
    is_qwen32 = model == "qwen32b"
    gated = "threshold" in variant or "roofline" in variant
    return {
        "timestamp": "2026-07-10T12:00:00Z",
        "model": model,
        "variant": variant,
        "gate_mode": (
            "roofline" if "roofline" in variant else "threshold" if gated else "off"
        ),
        "k": "0" if variant.startswith("baseline_") else "5",
        "threshold": "32" if gated else "",
        "consecutive_checks": "10" if gated else "",
        "roofline_config_sha256": "cafebabe" if "roofline" in variant else "",
        "cluster": "lyris-gb200",
        "runtime": "nemo-rl",
        "runtime_version": "nightly-20260707",
        "runtime_commit": "abc123",
        "vllm_version": "0.24.0",
        "vllm_commit": "ee0da84a",
        "target_tp": "2" if is_qwen32 else "1",
        "draft_tp": "1",
        "dp": "8" if is_qwen32 else "16",
        "ep": "1",
        "temperature": "1.0",
        "top_p": "1.0",
        "max_osl": "4096",
        "max_model_len": "4128",
        "max_sequence_length": "4096",
        "num_prompts": "64",
        "num_generations": "32",
        "train_gbs": "512",
        "max_num_batched_tokens": "16384",
        "max_num_seqs": "1024",
        "recipe": (
            "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
            if is_qwen32
            else "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
        ),
        "container": "/containers/nemo.sqsh",
        "container_sha256": "deadbeef",
        "runner": runner,
        "graph_mode": "FULL_AND_PIECEWISE" if runner == "v2" else "PIECEWISE",
        "sampling": "standard",
        "job_id": f"job-{model}-{variant}",
        "wandb_run_id": f"run-{model}-{variant}",
        "wandb_url": f"https://wandb.example/{model}/{variant}",
        "command": f"run --model={model} --variant={variant}",
    }


def _is_specdec(metadata: Mapping[str, str]) -> bool:
    return not metadata["variant"].startswith("baseline_")


def _is_gated(metadata: Mapping[str, str]) -> bool:
    return metadata["gate_mode"] in {"threshold", "roofline"}


def _history(
    metadata: Mapping[str, str],
    *,
    scale: float = 1.0,
    target_graph_ratio: float = 1.0,
    draft_graph_ratio: float = 1.0,
    activated: bool = True,
    predicted_speedup: float = 1.12,
    activation_predicted_speedup: float = 1.12,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for step in range(1, 21):
        row = {
            "_step": step,
            "timing/train/total_step_time": 200.0 / scale,
            "timing/train/generation": 100.0 / scale,
            "performance/tokens_per_sec_per_gpu": 25.0 * scale,
            "performance/generation_tokens_per_sec_per_gpu": 50.0 * scale,
            "timing/train/policy_training": 30.0,
            "timing/train/policy_and_reference_logprobs": 20.0,
            "train/reward": 0.4,
            "train/mean_gen_tokens_per_sample": 1024.0,
            "train/gen_kl_error": 0.01,
            "train/loss": 0.2,
            "train/vllm/cudagraph_target_graph_call_ratio": target_graph_ratio,
        }
        if _is_specdec(metadata):
            row.update(
                {
                    "train/vllm/spec_num_drafts": 100.0,
                    "train/vllm/spec_num_draft_tokens": 300.0,
                    "train/vllm/spec_num_accepted_tokens": 150.0,
                    "train/vllm/spec_acceptance_rate": 0.5,
                    "train/vllm/spec_acceptance_length": 2.5,
                }
            )
            if metadata["runner"] == "v1":
                row["train/vllm/cudagraph_draft_graph_call_ratio"] = draft_graph_ratio
            else:
                row["train/vllm/cudagraph_draft_prefill_graph_call_ratio"] = (
                    draft_graph_ratio
                )
                row["train/vllm/cudagraph_draft_decode_graph_call_ratio"] = (
                    draft_graph_ratio
                )
        if _is_gated(metadata):
            row.update(
                {
                    "train/vllm/tail_gate_decisions": 100.0,
                    "train/vllm/tail_gate_activations": 1.0 if activated else 0.0,
                    "train/vllm/tail_gate_enabled_step_ratio": (
                        0.25 if activated else 0.0
                    ),
                    "train/vllm/tail_gate_advance_only_step_ratio": (
                        0.75 if activated else 1.0
                    ),
                    "train/vllm/tail_gate_activation_tick": (
                        17.0 if activated else 0.0
                    ),
                    "train/vllm/tail_gate_activation_batch": (
                        16.0 if activated else 0.0
                    ),
                    "train/vllm/tail_gate_activation_seq_len": (
                        8192.0 if activated else 0.0
                    ),
                    "train/vllm/tail_gate_predicted_speedup": (
                        predicted_speedup if activated else 0.0
                    ),
                }
            )
            if metadata["gate_mode"] == "roofline":
                row["train/vllm/tail_gate_activation_predicted_speedup"] = (
                    activation_predicted_speedup if activated else 0.0
                )
        rows.append(row)
    return rows


def _summary(
    metadata: dict[str, str],
    *,
    scale: float = 1.0,
    target_graph_ratio: float = 1.0,
    draft_graph_ratio: float = 1.0,
    activated: bool = True,
    predicted_speedup: float = 1.12,
    activation_predicted_speedup: float = 1.12,
) -> RunSummary:
    return summarize_history(
        metadata,
        _history(
            metadata,
            scale=scale,
            target_graph_ratio=target_graph_ratio,
            draft_graph_ratio=draft_graph_ratio,
            activated=activated,
            predicted_speedup=predicted_speedup,
            activation_predicted_speedup=activation_predicted_speedup,
        ),
    )


def _cohort(*, model: str = "qwen32b", runner: str = "v2") -> list[dict[str, str]]:
    variants = (
        ["baseline_v1", "always_on_v1_k5"]
        if runner == "v1"
        else ["baseline_v2", "always_on_v2_k5", "fastrl_threshold_v2_k5"]
    )
    return [
        _metadata(model=model, runner=runner, variant=variant) for variant in variants
    ]


def _write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    materialized = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, delimiter="\t", fieldnames=list(materialized[0])
        )
        writer.writeheader()
        writer.writerows(materialized)


class _FakeRun:
    def __init__(self, history: list[dict[str, float]], url: str) -> None:
        self._history = history
        self.url = url
        self.requested_keys: list[str] = []

    def scan_history(self, *, keys: list[str]):
        self.requested_keys = keys
        return iter(
            {key: row[key] for key in keys if key in row} for row in self._history
        )


class _FakeApi:
    def __init__(self, histories: Mapping[str, list[dict[str, float]]]) -> None:
        self._histories = histories
        self.runs: dict[str, _FakeRun] = {}
        self.calls = 0

    def run(self, path: str) -> _FakeRun:
        self.calls += 1
        run_id = path.rsplit("/", maxsplit=1)[-1]
        run = _FakeRun(self._histories[run_id], f"https://wandb.example/api/{run_id}")
        self.runs[run_id] = run
        return run


def _api_for(rows: Iterable[dict[str, str]]) -> _FakeApi:
    return _FakeApi(
        {
            row["wandb_run_id"]: _history(
                row,
                scale=1.25 if not row["variant"].startswith("baseline_") else 1.0,
            )
            for row in rows
        }
    )


def test_metric_contract_uses_exact_production_emitter_keys() -> None:
    assert METRIC_KEYS == EXPECTED_METRIC_KEYS


def test_history_keys_are_variant_and_runner_aware() -> None:
    baseline = _metadata()
    v1_spec = _metadata(model="qwen30ba3b", runner="v1", variant="always_on_v1_k5")
    gated = _metadata(variant="fastrl_threshold_v2_k5")
    roofline = _metadata(variant="efficient_roofline_v2_k5")

    baseline_keys = _history_keys(baseline)
    assert "train/vllm/cudagraph_target_graph_call_ratio" in baseline_keys
    assert not any("spec_" in key or "tail_gate_" in key for key in baseline_keys)
    assert not any("cudagraph_draft" in key for key in baseline_keys)

    v1_keys = _history_keys(v1_spec)
    assert "train/vllm/spec_num_drafts" in v1_keys
    assert "train/vllm/cudagraph_draft_graph_call_ratio" in v1_keys
    assert not any("tail_gate_" in key for key in v1_keys)

    gated_keys = _history_keys(gated)
    assert "train/vllm/tail_gate_activations" in gated_keys
    assert "train/vllm/tail_gate_activation_predicted_speedup" not in gated_keys
    assert "train/vllm/cudagraph_draft_prefill_graph_call_ratio" in gated_keys
    assert "train/vllm/cudagraph_draft_decode_graph_call_ratio" in gated_keys

    assert "train/vllm/tail_gate_activation_predicted_speedup" in _history_keys(
        roofline
    )


def test_baseline_is_final_without_specdec_gate_or_draft_metrics() -> None:
    summary = _summary(_metadata())

    assert summary.status == "final"
    assert summary.steps == list(range(2, 21))
    assert summary.policy_time == 30.0
    assert summary.logprob_time == 20.0
    assert summary.policy_loss == 0.2
    assert summary.acceptance_rate is None
    assert summary.draft_prefill_graph_ratio is None


@pytest.mark.parametrize(
    ("variant", "missing_key", "reason_metric"),
    [
        ("always_on_v2_k5", "train/vllm/spec_num_drafts", "num_drafts"),
        (
            "fastrl_threshold_v2_k5",
            "train/vllm/tail_gate_activations",
            "gate_activations",
        ),
        (
            "always_on_v2_k5",
            "train/vllm/cudagraph_draft_decode_graph_call_ratio",
            "draft_decode_graph_ratio",
        ),
    ],
)
def test_variant_required_metrics_cannot_be_missing(
    variant: str, missing_key: str, reason_metric: str
) -> None:
    metadata = _metadata(variant=variant)
    history = _history(metadata)
    del history[5][missing_key]

    summary = summarize_history(metadata, history)

    assert summary.status == "partial"
    assert summary.reason == f"missing_metric:{reason_metric}:6"


def test_activation_tick_is_required_only_for_gated_variants() -> None:
    gated = _metadata(variant="fastrl_threshold_v2_k5")
    baseline = _metadata()
    history = _history(gated)
    del history[5]["train/vllm/tail_gate_activation_tick"]

    summary = summarize_history(gated, history)

    assert summary.status == "partial"
    assert summary.reason == "missing_metric:activation_tick:6"
    assert "train/vllm/tail_gate_activation_tick" in _history_keys(gated)
    assert "train/vllm/tail_gate_activation_tick" not in _history_keys(baseline)


def test_nonfinite_production_metric_is_partial() -> None:
    metadata = _metadata()
    history = _history(metadata)
    history[8]["train/loss"] = math.nan

    summary = summarize_history(metadata, history)

    assert summary.status == "partial"
    assert summary.reason == "non_finite_metric:policy_loss:9"


def test_comparison_key_uses_only_complete_explicit_cohort_schema() -> None:
    baseline_metadata = _metadata()
    candidate_metadata = _metadata(variant="always_on_v2_k5")
    candidate_metadata["command"] = "a deliberately different command string"

    rows = build_comparison_rows(
        [_summary(baseline_metadata), _summary(candidate_metadata, scale=1.25)]
    )

    candidate = next(row for row in rows if row.variant == "always_on_v2_k5")
    assert candidate.e2e_time_speedup_vs_baseline == 1.25


@pytest.mark.parametrize(
    "field",
    [
        "cluster",
        "runtime_version",
        "runtime_commit",
        "vllm_version",
        "vllm_commit",
        "target_tp",
        "dp",
        "temperature",
        "max_osl",
        "max_model_len",
        "num_generations",
        "graph_mode",
        "sampling",
    ],
)
def test_cross_cohort_speedups_are_rejected(field: str) -> None:
    baseline = _metadata()
    candidate = _metadata(variant="always_on_v2_k5")
    candidate[field] = f"different-{candidate[field]}"

    with pytest.raises(ValueError, match="missing matched baseline"):
        build_comparison_rows([_summary(baseline), _summary(candidate)])


def test_manifest_rejects_every_missing_cohort_dimension() -> None:
    row = _metadata()

    for field in COHORT_FIELDS:
        incomplete = {**row, field: ""}
        assert (
            _validate_manifest_rows([incomplete]) == f"missing manifest fields:{field}"
        )


def test_manifest_rejects_engine_length_below_output_plus_headroom() -> None:
    row = _metadata()
    row["max_model_len"] = "4096"

    assert _validate_manifest_rows([row]) == (
        "max_model_len must be at least max_osl plus 32:4096:4096"
    )


@pytest.mark.parametrize(
    ("target_graph_ratio", "draft_graph_ratio"), [(0.98, 1.0), (1.0, 0.98)]
)
def test_graph_coverage_below_threshold_is_health_failed(
    target_graph_ratio: float, draft_graph_ratio: float
) -> None:
    metadata = _cohort()
    summaries = [
        _summary(
            row,
            target_graph_ratio=(
                target_graph_ratio
                if row["variant"] == "fastrl_threshold_v2_k5"
                else 1.0
            ),
            draft_graph_ratio=(
                draft_graph_ratio if row["variant"] == "fastrl_threshold_v2_k5" else 1.0
            ),
        )
        for row in metadata
    ]

    candidate = next(
        row
        for row in build_comparison_rows(summaries)
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert candidate.cuda_graph_health_passed is False
    assert candidate.health_gate_passed is False
    assert candidate.status == "health_failed"


def test_gated_variant_must_activate_after_observable_off_period() -> None:
    metadata = _cohort()
    summaries = [
        _summary(row, activated=row["variant"] != "fastrl_threshold_v2_k5")
        for row in metadata
    ]

    candidate = next(
        row
        for row in build_comparison_rows(summaries)
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert candidate.gate_activation_health_passed is False
    assert candidate.health_gate_passed is False
    assert candidate.status == "health_failed"


def test_gated_variant_cannot_report_always_on_behavior() -> None:
    cohort = _cohort()
    gated_metadata = cohort[-1]
    gated_history = _history(gated_metadata)
    for record in gated_history:
        record["train/vllm/tail_gate_enabled_step_ratio"] = 1.0
        record["train/vllm/tail_gate_advance_only_step_ratio"] = 0.0
    summaries = [
        _summary(cohort[0]),
        _summary(cohort[1]),
        summarize_history(gated_metadata, gated_history),
    ]

    candidate = next(
        row
        for row in build_comparison_rows(summaries)
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert candidate.gate_activation_health_passed is False
    assert candidate.status == "health_failed"


def test_roofline_requires_explicit_activation_predicted_speedup() -> None:
    metadata = _metadata(variant="efficient_roofline_v2_k5")
    history = _history(metadata)
    del history[5]["train/vllm/tail_gate_activation_predicted_speedup"]

    summary = summarize_history(metadata, history)

    assert summary.status == "partial"
    assert summary.reason == "missing_metric:activation_predicted_speedup:6"


@pytest.mark.parametrize(
    ("all_decision_speedup", "activation_speedup", "expected_status"),
    [(1.30, 1.01, "health_failed"), (0.90, 1.06, "final")],
)
def test_roofline_health_uses_only_activation_predicted_speedup(
    all_decision_speedup: float,
    activation_speedup: float,
    expected_status: str,
) -> None:
    baseline = _metadata()
    always_on = _metadata(variant="always_on_v2_k5")
    roofline = _metadata(variant="efficient_roofline_v2_k5")
    rows = build_comparison_rows(
        [
            _summary(baseline),
            _summary(always_on),
            _summary(
                roofline,
                predicted_speedup=all_decision_speedup,
                activation_predicted_speedup=activation_speedup,
            ),
        ]
    )

    candidate = next(row for row in rows if row.variant == "efficient_roofline_v2_k5")

    assert candidate.predicted_speedup == all_decision_speedup
    assert candidate.activation_predicted_speedup == activation_speedup
    assert candidate.status == expected_status


def test_main_renders_interleaved_models_and_runner_sections(tmp_path: Path) -> None:
    rows = [
        *_cohort(model="qwen32b", runner="v2"),
        *_cohort(model="qwen30ba3b", runner="v1"),
    ]
    interleaved = [rows[3], rows[0], rows[4], rows[2], rows[1]]
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, interleaved)
    output_dir = tmp_path / "output"

    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_api_for(rows),
        )
        == 0
    )

    fragment = (output_dir / "tail_gated_specdec.html").read_text(encoding="utf-8")
    assert "Model Runner V1" in fragment
    assert "Model Runner V2" in fragment
    assert "qwen30ba3b" in fragment
    assert "qwen32b" in fragment
    assert "Final finding:" in fragment


def test_findings_exclude_health_failed_rows(tmp_path: Path) -> None:
    rows = _cohort()
    histories = {
        row["wandb_run_id"]: _history(
            row,
            scale=2.0 if row["variant"] == "fastrl_threshold_v2_k5" else 1.0,
            activated=row["variant"] != "fastrl_threshold_v2_k5",
        )
        for row in rows
    }
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    output_dir = tmp_path / "output"

    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_FakeApi(histories),
        )
        == 1
    )

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    failed = next(row for row in payload if row["variant"] == "fastrl_threshold_v2_k5")
    assert failed["status"] == "health_failed"
    finding = next(
        line
        for line in (output_dir / "tail_gated_specdec.html")
        .read_text(encoding="utf-8")
        .splitlines()
        if "Final finding:" in line
    )
    assert "fastrl_threshold_v2_k5" not in finding


def test_output_directory_is_claimed_before_wandb_fetch(tmp_path: Path) -> None:
    rows = _cohort()
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    output_dir = tmp_path / "already-claimed"
    output_dir.mkdir()
    api = _api_for(rows)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == 0


def test_shuffled_manifest_produces_byte_identical_artifacts(tmp_path: Path) -> None:
    rows = [
        *_cohort(model="qwen32b", runner="v2"),
        *_cohort(model="qwen30ba3b", runner="v1"),
    ]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_manifest = first_dir / "submissions.tsv"
    second_manifest = second_dir / "submissions.tsv"
    _write_manifest(first_manifest, rows)
    _write_manifest(second_manifest, reversed(rows))

    assert (
        main(
            [
                "--manifest",
                str(first_manifest),
                "--output-dir",
                str(first_dir / "output"),
            ],
            api=_api_for(rows),
        )
        == 0
    )
    assert (
        main(
            [
                "--manifest",
                str(second_manifest),
                "--output-dir",
                str(second_dir / "output"),
            ],
            api=_api_for(rows),
        )
        == 0
    )

    for filename in ("summary.csv", "summary.json", "tail_gated_specdec.html"):
        assert (first_dir / "output" / filename).read_bytes() == (
            second_dir / "output" / filename
        ).read_bytes()


def test_output_rows_include_full_metric_speedup_health_and_provenance_contract() -> (
    None
):
    rows = build_comparison_rows([_summary(row) for row in _cohort()])

    assert set(REQUIRED_ROW_FIELDS) == set(rows[0].to_dict())
    assert set(COHORT_FIELDS).issubset(rows[0].to_dict())
