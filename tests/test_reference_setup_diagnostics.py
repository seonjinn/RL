import faulthandler
import json
from datetime import timedelta

from nemo_rl.models.megatron import reference_setup_diagnostics as diagnostics
from nemo_rl.models.megatron.reference_setup_diagnostics import (
    _cgroup_memory_metadata,
    _numa_memory_metadata,
    buffer_memory_metadata,
    checkpoint_marker_metadata,
    distributed_timeout_override,
    log_reference_setup_stage,
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


def test_numa_memory_metadata_reports_each_cpu_node(tmp_path) -> None:
    node_root = tmp_path / "node"
    for node_id, free_kb, used_kb in ((0, 420_000, 50_000), (1, 310_000, 160_000)):
        node_dir = node_root / f"node{node_id}"
        node_dir.mkdir(parents=True)
        (node_dir / "meminfo").write_text(
            f"Node {node_id} MemTotal: 470000 kB\n"
            f"Node {node_id} MemFree: {free_kb} kB\n"
            f"Node {node_id} MemUsed: {used_kb} kB\n"
        )

    assert _numa_memory_metadata(node_root) == {
        "numa_0_mem_free_kb": 420_000,
        "numa_0_mem_total_kb": 470_000,
        "numa_0_mem_used_kb": 50_000,
        "numa_1_mem_free_kb": 310_000,
        "numa_1_mem_total_kb": 470_000,
        "numa_1_mem_used_kb": 160_000,
    }


def test_cgroup_memory_metadata_reports_limits_events_and_numa_usage(tmp_path) -> None:
    cgroup_root = tmp_path / "cgroup"
    cgroup_path = cgroup_root / "slurm" / "job-42"
    cgroup_path.mkdir(parents=True)
    proc_self_cgroup = tmp_path / "self.cgroup"
    proc_self_cgroup.write_text("0::/slurm/job-42\n")
    (cgroup_path / "memory.current").write_text("123456\n")
    (cgroup_path / "memory.peak").write_text("234567\n")
    (cgroup_path / "memory.max").write_text("987654\n")
    (cgroup_path / "memory.events").write_text(
        "low 0\nhigh 4\nmax 7\noom 2\noom_kill 1\n"
    )
    (cgroup_path / "memory.numa_stat").write_text(
        "anon N0=1000 N1=2000\nfile N0=3000 N1=4000\n"
    )

    assert _cgroup_memory_metadata(proc_self_cgroup, cgroup_root) == {
        "cgroup_memory_current_bytes": 123456,
        "cgroup_memory_event_high": 4,
        "cgroup_memory_event_max": 7,
        "cgroup_memory_event_oom": 2,
        "cgroup_memory_event_oom_kill": 1,
        "cgroup_memory_max_bytes": 987654,
        "cgroup_memory_numa_anon_n0_bytes": 1000,
        "cgroup_memory_numa_anon_n1_bytes": 2000,
        "cgroup_memory_numa_file_n0_bytes": 3000,
        "cgroup_memory_numa_file_n1_bytes": 4000,
        "cgroup_memory_peak_bytes": 234567,
    }


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
