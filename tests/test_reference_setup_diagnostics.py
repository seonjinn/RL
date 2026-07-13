import faulthandler
import json
from datetime import timedelta

from nemo_rl.models.megatron import reference_setup_diagnostics as diagnostics
from nemo_rl.models.megatron.reference_setup_diagnostics import (
    buffer_memory_metadata,
    checkpoint_marker_metadata,
    distributed_timeout_override,
    log_reference_setup_stage,
    reference_cpu_offload_lock,
    reference_cpu_offload_serialization_enabled,
    reference_setup_stack_dumps,
)


class _FakeTensor:
    def __init__(self, numel: int, element_size: int) -> None:
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


def test_reference_setup_diagnostics_are_disabled_by_default(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("NRL_DEBUG_REFERENCE_MODEL_SETUP", raising=False)

    log_reference_setup_stage("setup.enter")

    assert capsys.readouterr().err == ""


def test_reference_setup_stack_dump_timer_is_disabled_by_default(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.delenv("NRL_DEBUG_REFERENCE_MODEL_SETUP", raising=False)
    monkeypatch.setenv("NRL_REFERENCE_SETUP_STACK_DUMP_SECONDS", "600")
    monkeypatch.setattr(
        diagnostics.threading,
        "Thread",
        lambda *args, **kwargs: calls.append("start"),
    )

    with reference_setup_stack_dumps():
        calls.append("body")

    assert calls == ["body"]


def test_reference_setup_stage_includes_rank_and_process_metadata(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("NRL_DEBUG_REFERENCE_MODEL_SETUP", "1")
    monkeypatch.setenv("RANK", "17")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "64")

    log_reference_setup_stage("setup.before_get_model")

    output = capsys.readouterr().err
    assert "NRL_REFERENCE_SETUP" in output
    assert "stage=setup.before_get_model" in output
    assert "rank=17" in output
    assert "local_rank=1" in output
    assert "world_size=64" in output
    assert "pid=" in output
    assert "host=" in output
    assert "epoch_s=" in output


def test_reference_setup_stage_writes_rank_local_jsonl_marker(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("NRL_DEBUG_REFERENCE_MODEL_SETUP", "1")
    monkeypatch.setenv("NRL_REFERENCE_SETUP_MARKER_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "17")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "64")

    log_reference_setup_stage(
        "worker.before_buffer_offload",
        buffer_group="dense",
        buffer_index=3,
    )

    marker_path = tmp_path / "rank-17.jsonl"
    marker = json.loads(marker_path.read_text().strip())
    assert marker["stage"] == "worker.before_buffer_offload"
    assert marker["rank"] == "17"
    assert marker["buffer_group"] == "dense"
    assert marker["buffer_index"] == 3
    assert marker["pid"] > 0


def test_buffer_memory_metadata_reports_param_grad_and_cpu_bytes() -> None:
    buffer = type(
        "FakeBuffer",
        (),
        {
            "param_data": _FakeTensor(10, 2),
            "grad_data": _FakeTensor(12, 4),
            "param_data_cpu": _FakeTensor(8, 2),
        },
    )()

    assert buffer_memory_metadata(buffer) == {
        "grad_bytes": 48,
        "grad_numel": 12,
        "param_bytes": 20,
        "param_cpu_bytes": 16,
        "param_cpu_numel": 8,
        "param_numel": 10,
    }


def test_reference_cpu_offload_serialization_is_disabled_by_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NRL_SERIALIZE_REFERENCE_CPU_OFFLOAD", raising=False)

    assert reference_cpu_offload_serialization_enabled() is False


def test_reference_cpu_offload_lock_serializes_when_enabled(
    monkeypatch, tmp_path
) -> None:
    calls: list[object] = []
    monkeypatch.setenv("NRL_SERIALIZE_REFERENCE_CPU_OFFLOAD", "1")
    monkeypatch.setenv("NRL_REFERENCE_CPU_OFFLOAD_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(
        diagnostics.fcntl,
        "flock",
        lambda _file, operation: calls.append(operation),
    )

    with reference_cpu_offload_lock(
        enabled=reference_cpu_offload_serialization_enabled()
    ):
        calls.append("body")

    assert calls == [diagnostics.fcntl.LOCK_EX, "body", diagnostics.fcntl.LOCK_UN]
    assert len(list(tmp_path.glob("*.lock"))) == 1


def test_reference_setup_stack_dump_worker_repeats_until_stopped(monkeypatch) -> None:
    calls: list[tuple] = []

    class StopAfterOneDump:
        def __init__(self) -> None:
            self.wait_count = 0

        def wait(self, timeout: int) -> bool:
            calls.append(("wait", timeout))
            self.wait_count += 1
            return self.wait_count > 1

    monkeypatch.setattr(
        faulthandler,
        "dump_traceback",
        lambda *, file, all_threads: calls.append(("dump", file, all_threads)),
    )

    diagnostics._dump_stacks_periodically(StopAfterOneDump(), 600)

    assert calls[0] == ("wait", 600)
    assert calls[1][0] == "dump"
    assert calls[1][2] is True
    assert calls[2] == ("wait", 600)


def test_reference_setup_stack_dump_thread_is_started_and_stopped(
    monkeypatch,
) -> None:
    calls: list[tuple] = []

    class FakeEvent:
        def set(self) -> None:
            calls.append(("set",))

    class FakeThread:
        def __init__(self, **kwargs) -> None:
            calls.append(("init", kwargs))

        def start(self) -> None:
            calls.append(("start",))

        def join(self, timeout: float) -> None:
            calls.append(("join", timeout))

    monkeypatch.setenv("NRL_DEBUG_REFERENCE_MODEL_SETUP", "1")
    monkeypatch.setenv("NRL_REFERENCE_SETUP_STACK_DUMP_SECONDS", "600")
    monkeypatch.setattr(diagnostics.threading, "Event", FakeEvent)
    monkeypatch.setattr(diagnostics.threading, "Thread", FakeThread)

    with reference_setup_stack_dumps():
        calls.append(("body",))

    assert calls[0][0] == "init"
    assert calls[0][1]["daemon"] is True
    assert calls[1] == ("start",)
    assert calls[2] == ("body",)
    assert calls[3] == ("set",)
    assert calls[4] == ("join", 1.0)


def test_reference_setup_stack_dump_timer_ignores_non_positive_interval(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setenv("NRL_DEBUG_REFERENCE_MODEL_SETUP", "1")
    monkeypatch.setenv("NRL_REFERENCE_SETUP_STACK_DUMP_SECONDS", "0")
    monkeypatch.setattr(
        diagnostics.threading,
        "Thread",
        lambda *args, **kwargs: calls.append("start"),
    )

    with reference_setup_stack_dumps():
        calls.append("body")

    assert calls == ["body"]


def test_distributed_timeout_override_is_unset_by_default(monkeypatch) -> None:
    monkeypatch.delenv("NRL_MEGATRON_NCCL_TIMEOUT_SECONDS", raising=False)

    assert distributed_timeout_override() is None


def test_distributed_timeout_override_uses_seconds(monkeypatch) -> None:
    monkeypatch.setenv("NRL_MEGATRON_NCCL_TIMEOUT_SECONDS", "1800")

    assert distributed_timeout_override() == timedelta(seconds=1800)


def test_checkpoint_marker_metadata_records_visible_and_missing_files(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("NRL_DEBUG_REFERENCE_MODEL_SETUP", "1")
    marker = tmp_path / "metadata.json"
    marker.write_text("{}")

    metadata = checkpoint_marker_metadata(tmp_path)

    assert metadata["realpath"] == str(tmp_path.resolve())
    assert metadata["metadata.json"]["exists"] is True
    assert metadata["metadata.json"]["size"] == 2
    assert metadata["run_config.yaml"]["exists"] is False
