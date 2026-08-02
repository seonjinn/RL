from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
MODULE_PATH = EXPERIMENT_DIR / "export_tensorboard.py"

CANONICAL_TAGS = (
    "timing/train/total_step_time",
    "timing/train/generation",
    "timing/train/policy_training",
    "timing/train/policy_and_reference_logprobs",
    "performance/tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu",
    "performance/policy_training_tokens_per_sec_per_gpu",
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    "cuda_graph/capture_count",
    "cuda_graph/replay_count",
    "cuda_graph/cache_hits",
    "cuda_graph/cache_misses",
    "cuda_graph/cache_evictions",
    "cuda_graph/fallback_count",
    "cuda_graph/graph_calls",
    "cuda_graph/eligible_calls",
    "cuda_graph/logical_tokens",
    "cuda_graph/padded_tokens",
    "cuda_graph/capacity_tokens",
    "cuda_graph/coverage",
    "cuda_graph/capacity_utilization",
    "cuda_graph/padding_utilization",
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
PROVENANCE = {
    "nemo_rl_commit": "1" * 40,
    "bridge_commit": "2" * 40,
    "mcore_commit": "3" * 40,
    "te_commit": "4" * 40,
    "te_version": "2.16.0.dev0",
    "container_sha256": "5" * 64,
}


def _load_exporter() -> ModuleType:
    pytest.importorskip("tensorboard")
    spec = importlib.util.spec_from_file_location("export_tensorboard", MODULE_PATH)
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


def _full_event_values(steps: int) -> dict[int, dict[str, float]]:
    return {
        step: {
            tag: float(index * 1000 + step)
            for index, tag in enumerate(CANONICAL_TAGS, start=1)
        }
        for step in range(1, steps + 1)
    }


@pytest.mark.parametrize("steps", [5, 20, 100])
def test_exporter_supports_every_planned_step_count(
    tmp_path: Path,
    steps: int,
) -> None:
    exporter = _load_exporter()
    events = tmp_path / "events"
    output = tmp_path / "results.jsonl"
    values = _full_event_values(steps)
    for metrics in values.values():
        metrics["train/accuracy"] = metrics.pop("train/reward")
    _write_events(
        events,
        suffix=".complete",
        values_by_step=values,
        wall_time_offset=100.0,
    )

    exporter.export_events(
        [events],
        model="nano",
        dispatcher="alltoall",
        scope="attn,moe_router",
        mode="nemorl",
        cluster="oci-hsg",
        profile="oci-hsg-gb200",
        phase="performance",
        steps=steps,
        repeat=1,
        run_group="nano-performance",
        job_id="2474000",
        status="passed",
        provenance=PROVENANCE,
        parity=None,
        output=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(1, steps + 1))
    assert all(row["steps"] == steps for row in rows)
    assert all(row["model"] == "nano" for row in rows)
    assert all(row["dispatcher"] == "alltoall" for row in rows)
    assert all(row["scope"] == "attn,moe_router" for row in rows)
    assert all(row["router_replay"] == "off" for row in rows)
    assert set(rows[0]["metrics"]) == set(CANONICAL_TAGS)
    assert "train/accuracy" not in rows[0]["metrics"]


def test_exporter_uses_latest_event_for_duplicate_step(tmp_path: Path) -> None:
    exporter = _load_exporter()
    events = tmp_path / "events"
    output = tmp_path / "results.jsonl"
    _write_events(
        events,
        suffix=".initial",
        values_by_step=_full_event_values(5),
        wall_time_offset=100.0,
    )
    _write_events(
        events,
        suffix=".retry",
        values_by_step={3: {"train/reward": 999.0}},
        wall_time_offset=200.0,
    )

    exporter.export_events(
        [events],
        model="nano",
        dispatcher="alltoall",
        scope="attn",
        mode="nemorl",
        cluster="oci-hsg",
        profile="oci-hsg-gb200",
        phase="smoke",
        steps=5,
        repeat=1,
        run_group="nano-smoke",
        job_id="2474000",
        status="passed",
        provenance=PROVENANCE,
        parity=None,
        output=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[2]["metrics"]["train/reward"] == 999.0


def test_exporter_preserves_router_replay_and_rejects_invalid_values(
    tmp_path: Path,
) -> None:
    exporter = _load_exporter()
    events = tmp_path / "events"
    output = tmp_path / "results.jsonl"
    _write_events(
        events,
        suffix=".complete",
        values_by_step=_full_event_values(5),
        wall_time_offset=100.0,
    )

    exporter.export_events(
        [events],
        model="nano",
        dispatcher="alltoall",
        scope="attn",
        mode="nemorl",
        cluster="oci-hsg",
        profile="oci-hsg-gb200",
        phase="performance",
        steps=5,
        repeat=1,
        run_group="nano-performance",
        job_id="2474000",
        status="passed",
        router_replay="on",
        provenance=PROVENANCE,
        parity=None,
        output=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["router_replay"] for row in rows} == {"on"}

    with pytest.raises(ValueError, match="router_replay must be one of off, on"):
        exporter.export_events(
            [events],
            model="nano",
            dispatcher="alltoall",
            scope="attn",
            mode="nemorl",
            cluster="oci-hsg",
            profile="oci-hsg-gb200",
            phase="performance",
            steps=5,
            repeat=1,
            run_group="nano-performance",
            job_id="2474000",
            status="passed",
            router_replay="enabled",
            provenance=PROVENANCE,
            parity=None,
            output=output,
        )


def test_exporter_reports_absent_required_tag_and_preserves_output(
    tmp_path: Path,
) -> None:
    exporter = _load_exporter()
    events = tmp_path / "events"
    output = tmp_path / "results.jsonl"
    output.write_text('{"previous":true}\n')
    incomplete_values = _full_event_values(5)
    for metrics in incomplete_values.values():
        del metrics["cuda_graph/eligible_calls"]
    _write_events(
        events,
        suffix=".incomplete",
        values_by_step=incomplete_values,
        wall_time_offset=100.0,
    )

    with pytest.raises(ValueError, match=r"cuda_graph/eligible_calls: missing_steps"):
        exporter.export_events(
            [events],
            model="nano",
            dispatcher="alltoall",
            scope="attn",
            mode="nemorl",
            cluster="oci-hsg",
            profile="oci-hsg-gb200",
            phase="smoke",
            steps=5,
            repeat=1,
            run_group="nano-smoke",
            job_id="2474000",
            status="failed",
            provenance=PROVENANCE,
            parity=None,
            output=output,
        )

    assert output.read_text() == '{"previous":true}\n'


def test_exporter_rejects_unplanned_step_count_before_reading_events(
    tmp_path: Path,
) -> None:
    exporter = _load_exporter()

    with pytest.raises(ValueError, match="steps must be one of 5, 20, 100"):
        exporter.export_events(
            [tmp_path / "missing"],
            model="nano",
            dispatcher="alltoall",
            scope="attn",
            mode="nemorl",
            cluster="oci-hsg",
            profile="oci-hsg-gb200",
            phase="custom",
            steps=7,
            repeat=1,
            run_group="nano-custom",
            job_id="2474000",
            status="failed",
            provenance=PROVENANCE,
            parity=None,
            output=tmp_path / "results.jsonl",
        )
