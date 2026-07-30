import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


EXPERIMENT_DIR = (
    Path(__file__).parents[3]
    / "experiments"
    / "cuda_graph"
    / "mamba_moe_te_graph_20260729"
)

CANONICAL_TAGS = (
    "performance/tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu",
    "performance/policy_training_tokens_per_sec_per_gpu",
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    "timing/train/total_step_time",
    "timing/train/generation",
    "timing/train/policy_training",
    "timing/train/policy_and_reference_logprobs",
    "train/reward",
    "train/gen_kl_error",
    "train/token_mult_prob_error",
    "train/policy_kl_error",
    "train/js_divergence_error",
    "train/sampling_importance_ratio",
    "train/num_masked_seqs_by_logprob_error",
    "train/loss",
    "train/grad_norm",
)


def _load_exporter() -> ModuleType:
    pytest.importorskip("tensorboard")
    path = EXPERIMENT_DIR / "export_tensorboard.py"
    spec = importlib.util.spec_from_file_location("export_tensorboard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXPERIMENT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _write_events(
    directory: Path,
    *,
    suffix: str,
    values_by_step: dict[int, dict[str, float]],
    wall_time_offset: float,
) -> None:
    tensorboard = pytest.importorskip("tensorboard")
    event_pb2 = tensorboard.compat.proto.event_pb2
    summary_pb2 = tensorboard.compat.proto.summary_pb2
    event_file_writer = tensorboard.summary.writer.event_file_writer

    writer = event_file_writer.EventFileWriter(str(directory), filename_suffix=suffix)
    try:
        for step, values in values_by_step.items():
            writer.add_event(
                event_pb2.Event(
                    wall_time=wall_time_offset + step,
                    step=step,
                    summary=summary_pb2.Summary(
                        value=[
                            summary_pb2.Summary.Value(tag=tag, simple_value=value)
                            for tag, value in values.items()
                        ]
                    ),
                )
            )
    finally:
        writer.close()


def _full_event_values() -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, float]] = {}
    for step in range(1, 21):
        values[step] = {
            tag: float(index * 100 + step)
            for index, tag in enumerate(CANONICAL_TAGS, start=1)
        }
        values[step]["train/accuracy"] = values[step].pop("train/reward")
    return values


def test_exporter_normalizes_alias_and_uses_latest_event_for_duplicate_step(
    tmp_path: Path,
) -> None:
    """A stale duplicate must not overwrite TensorBoard's newest scalar event."""
    exporter = _load_exporter()
    events = tmp_path / "events"
    output = tmp_path / "results.jsonl"
    _write_events(
        events,
        suffix=".initial",
        values_by_step=_full_event_values(),
        wall_time_offset=100.0,
    )
    _write_events(
        events,
        suffix=".retry",
        values_by_step={10: {"train/accuracy": 999.0}},
        wall_time_offset=200.0,
    )

    exporter.export_events(
        [events],
        scope="drop-pad-moe",
        job_id="2474000",
        status="performance:passed",
        output=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(1, 21))
    assert all(
        {
            "scope": "drop-pad-moe",
            "job_id": "2474000",
            "status": "performance:passed",
        }.items()
        <= row.items()
        for row in rows
    )
    assert rows[9]["metrics"]["train/reward"] == 999.0
    assert "train/accuracy" not in rows[9]["metrics"]
    assert set(rows[0]["metrics"]) == set(CANONICAL_TAGS)
    assert (
        rows[0]["metrics"]["train/gen_kl_error"]
        != rows[0]["metrics"]["train/token_mult_prob_error"]
    )


def test_exporter_reports_absent_required_tag_and_preserves_output(
    tmp_path: Path,
) -> None:
    """An incomplete event set must fail rather than publishing partial JSONL."""
    exporter = _load_exporter()
    events = tmp_path / "events"
    output = tmp_path / "results.jsonl"
    output.write_text('{"previous": true}\n')
    incomplete_values = _full_event_values()
    for metrics in incomplete_values.values():
        del metrics["train/policy_kl_error"]
    _write_events(
        events,
        suffix=".incomplete",
        values_by_step=incomplete_values,
        wall_time_offset=100.0,
    )

    with pytest.raises(ValueError) as error:
        exporter.export_events(
            [events],
            scope="drop-pad-moe",
            job_id="2474000",
            status="performance:passed",
            output=output,
        )

    assert (
        "train/policy_kl_error: missing_steps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "
        "12, 13, 14, 15, 16, 17, 18, 19, 20], count=0"
    ) in str(error.value)
    assert output.read_text() == '{"previous": true}\n'


def test_exporter_canonicalizes_legacy_masked_sequence_tag(tmp_path: Path) -> None:
    """A legacy real-event tag must populate the explicit canonical CSV field."""
    exporter = _load_exporter()
    events = tmp_path / "events"
    output = tmp_path / "results.jsonl"
    values = _full_event_values()
    for metrics in values.values():
        metrics["train/num_mask_sample_filtered"] = metrics.pop(
            "train/num_masked_seqs_by_logprob_error"
        )
    _write_events(
        events,
        suffix=".legacy-masked",
        values_by_step=values,
        wall_time_offset=100.0,
    )

    exporter.export_events(
        [events],
        scope="drop-pad-moe",
        job_id="2474000",
        status="performance:passed",
        output=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["metrics"]["train/num_masked_seqs_by_logprob_error"] == 1501.0
    assert "train/num_mask_sample_filtered" not in rows[0]["metrics"]
