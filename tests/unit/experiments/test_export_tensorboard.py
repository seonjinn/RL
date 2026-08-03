from __future__ import annotations

import base64
import builtins
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
IDENTITY = {
    "model": "nano",
    "dispatcher": "alltoall",
    "scope": "attn",
    "mode": "nemorl",
    "cluster": "oci-hsg",
    "profile": "oci-hsg-gb200",
    "phase": "smoke",
    "steps": "5",
    "repeat": "0",
    "run_group": "nano-smoke",
    "job_id": "2474000",
    "router_replay": "off",
}


def _write_run_metadata(
    path: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> None:
    values = {
        "schema_version": "1",
        **IDENTITY,
        "nemo_rl_commit": PROVENANCE["nemo_rl_commit"],
        "bridge_commit": PROVENANCE["bridge_commit"],
        "mcore_commit": PROVENANCE["mcore_commit"],
        "transformer_engine_commit": PROVENANCE["te_commit"],
        "container_sha256": PROVENANCE["container_sha256"],
        "effective_command_base64": base64.b64encode(b"echo exact command").decode(),
    }
    values.update(overrides or {})
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))


def _load_exporter() -> ModuleType:
    pytest.importorskip("tensorboard")
    spec = importlib.util.spec_from_file_location("export_tensorboard", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXPERIMENT_DIR))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        sys.path.pop(0)
    return module


def _write_events(
    directory: Path,
    *,
    suffix: str,
    values_by_step: dict[int, dict[str, float]],
    wall_time_offset: float,
) -> None:
    event_pb2 = pytest.importorskip("tensorboard.compat.proto.event_pb2")
    summary_pb2 = pytest.importorskip("tensorboard.compat.proto.summary_pb2")
    event_file_writer = pytest.importorskip(
        "tensorboard.summary.writer.event_file_writer"
    )

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


def _full_scalar_values(steps: int) -> dict[str, dict[int, float]]:
    rows = _full_event_values(steps)
    return {
        tag: {step: values[tag] for step, values in rows.items()}
        for tag in CANONICAL_TAGS
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
        router_replay="off",
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
        router_replay="off",
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


@pytest.mark.parametrize(
    "missing_tag", ("cuda_graph/eligible_calls", "cuda_graph/cache_misses")
)
def test_exporter_reports_absent_required_tag_and_preserves_output(
    tmp_path: Path,
    missing_tag: str,
) -> None:
    exporter = _load_exporter()
    events = tmp_path / "events"
    output = tmp_path / "results.jsonl"
    output.write_text('{"previous":true}\n')
    incomplete_values = _full_event_values(5)
    for metrics in incomplete_values.values():
        del metrics[missing_tag]
    _write_events(
        events,
        suffix=".incomplete",
        values_by_step=incomplete_values,
        wall_time_offset=100.0,
    )

    with pytest.raises(ValueError, match=rf"{missing_tag}: missing_steps"):
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
            router_replay="off",
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
            router_replay="off",
            provenance=PROVENANCE,
            parity=None,
            output=tmp_path / "results.jsonl",
        )


def test_module_import_does_not_require_tensorboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_tensorboard(name: str, *args: object, **kwargs: object) -> object:
        if name == "tensorboard" or name.startswith("tensorboard."):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_tensorboard)
    spec = importlib.util.spec_from_file_location(
        "export_tensorboard_lazy", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)


def test_exporter_accepts_task2_metadata_only_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = _load_exporter()
    metadata = tmp_path / "run-metadata.env"
    _write_run_metadata(metadata)
    events = tmp_path / "events"
    events.mkdir()
    monkeypatch.setattr(
        exporter, "_scalar_events", lambda paths: _full_scalar_values(5)
    )
    output = tmp_path / "results.jsonl"

    exporter.export_events(
        [events],
        run_metadata=metadata,
        status="passed",
        provenance=PROVENANCE,
        output=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 5
    assert {row["repeat"] for row in rows} == {0}
    assert {row["router_replay"] for row in rows} == {"off"}


def test_exporter_requires_explicit_or_metadata_router_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = _load_exporter()
    reads: list[object] = []
    monkeypatch.setattr(exporter, "_scalar_events", lambda paths: reads.append(paths))

    with pytest.raises(ValueError, match="launch identity is missing: router_replay"):
        exporter.export_events(
            [tmp_path / "missing"],
            model="nano",
            dispatcher="alltoall",
            scope="attn",
            mode="nemorl",
            cluster="oci-hsg",
            profile="oci-hsg-gb200",
            phase="smoke",
            steps=5,
            repeat=0,
            run_group="nano-smoke",
            job_id="2474000",
            status="failed",
            provenance=PROVENANCE,
            output=tmp_path / "results.jsonl",
        )
    assert reads == []


@pytest.mark.parametrize(
    ("field", "explicit"),
    (
        ("model", "super"),
        ("dispatcher", "hybridep"),
        ("scope", "mlp"),
        ("mode", "mcore"),
        ("cluster", "lyris"),
        ("profile", "alternate"),
        ("phase", "performance"),
        ("steps", 20),
        ("repeat", 1),
        ("run_group", "other-group"),
        ("job_id", "2474001"),
        ("router_replay", "on"),
    ),
)
def test_exporter_rejects_every_metadata_identity_mismatch_before_metric_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    explicit: str | int,
) -> None:
    exporter = _load_exporter()
    metadata = tmp_path / "run-metadata.env"
    _write_run_metadata(metadata)
    reads: list[object] = []
    monkeypatch.setattr(exporter, "_scalar_events", lambda paths: reads.append(paths))

    with pytest.raises(ValueError, match=rf"run metadata {field} disagrees"):
        exporter.export_events(
            [tmp_path / "missing"],
            run_metadata=metadata,
            status="failed",
            provenance=PROVENANCE,
            output=tmp_path / "results.jsonl",
            **{field: explicit},
        )
    assert reads == []


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"model=nano\nmodel=super\n",
        b"Model=nano\n",
        b"model nano\n",
        b"model=nano\r\n",
        b"model=nano\x00suffix\n",
        b"model='nano'\n",
        b"model=\\nano\n",
        b"model=$(touch-marker)\n",
        b"model=`touch-marker`\n",
        b"model=nano;touch-marker\n",
        b"model=nano|touch-marker\n",
        b"model=nano&touch-marker\n",
        b"model=nano>touch-marker\n",
        b"model= nano\n",
        b"\xff\n",
    ),
)
def test_run_metadata_parser_rejects_malformed_or_shell_active_input(
    tmp_path: Path,
    payload: bytes,
) -> None:
    exporter = _load_exporter()
    metadata = tmp_path / "run-metadata.env"
    marker = tmp_path / "touch-marker"
    metadata.write_bytes(payload.replace(b"touch-marker", str(marker).encode()))

    with pytest.raises(ValueError):
        exporter._read_run_metadata(metadata)
    assert not marker.exists()


def test_run_metadata_parser_rejects_symlink(tmp_path: Path) -> None:
    exporter = _load_exporter()
    target = tmp_path / "target.env"
    _write_run_metadata(target)
    metadata = tmp_path / "run-metadata.env"
    metadata.symlink_to(target)

    with pytest.raises(ValueError, match="safe regular file"):
        exporter._read_run_metadata(metadata)


def test_metadata_provenance_mismatch_precedes_metric_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = _load_exporter()
    metadata = tmp_path / "run-metadata.env"
    _write_run_metadata(metadata, overrides={"mcore_commit": "9" * 40})
    reads: list[object] = []
    monkeypatch.setattr(exporter, "_scalar_events", lambda paths: reads.append(paths))

    with pytest.raises(ValueError, match="mcore_commit disagrees with provenance"):
        exporter.export_events(
            [tmp_path / "missing"],
            run_metadata=metadata,
            status="failed",
            provenance=PROVENANCE,
            output=tmp_path / "results.jsonl",
        )
    assert reads == []
