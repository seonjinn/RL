from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.utils import host_memory


_GIB = 1024**3


def _configure_cgroup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    relative_path: str = "/slurm/job",
    use_mount_root: bool = False,
    current: str = str(2 * _GIB),
    maximum: str = str(4 * _GIB),
    peak: str = str(3 * _GIB),
) -> None:
    proc_self_cgroup = tmp_path / "proc-self-cgroup"
    proc_self_cgroup.write_text(f"0::{relative_path}\n")
    cgroup_root = tmp_path / "sys-fs-cgroup"
    cgroup_dir = (
        cgroup_root if use_mount_root else cgroup_root / relative_path.lstrip("/")
    )
    cgroup_dir.mkdir(parents=True)
    (cgroup_dir / "memory.current").write_text(current)
    (cgroup_dir / "memory.max").write_text(maximum)
    (cgroup_dir / "memory.peak").write_text(peak)
    monkeypatch.setattr(
        host_memory, "_PROC_SELF_CGROUP_PATH", proc_self_cgroup, raising=False
    )
    monkeypatch.setattr(host_memory, "_CGROUP_ROOT", cgroup_root, raising=False)


def _configure_psutil(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rss: int = 5 * _GIB,
    available: int = 10 * _GIB,
) -> None:
    monkeypatch.setattr(
        host_memory.psutil.Process,
        "memory_info",
        lambda _process: SimpleNamespace(rss=rss),
    )
    monkeypatch.setattr(
        host_memory.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=available),
    )


def test_host_memory_snapshot_reads_unified_cgroup_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup(monkeypatch, tmp_path)
    _configure_psutil(monkeypatch)

    snapshot = host_memory._get_host_memory_snapshot()

    assert snapshot.cgroup_memory_current_gib == 2.0
    assert snapshot.cgroup_memory_max_gib == 4.0
    assert snapshot.cgroup_memory_peak_gib == 3.0


def test_host_memory_snapshot_falls_back_to_cgroup_mount_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup(monkeypatch, tmp_path, use_mount_root=True)
    _configure_psutil(monkeypatch)

    snapshot = host_memory._get_host_memory_snapshot()

    assert snapshot.cgroup_memory_current_gib == 2.0
    assert snapshot.cgroup_memory_max_gib == 4.0
    assert snapshot.cgroup_memory_peak_gib == 3.0


def test_host_memory_snapshot_treats_unbounded_cgroup_max_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup(monkeypatch, tmp_path, maximum="max")
    _configure_psutil(monkeypatch)

    snapshot = host_memory._get_host_memory_snapshot()

    assert snapshot.cgroup_memory_current_gib == 2.0
    assert snapshot.cgroup_memory_max_gib is None
    assert snapshot.cgroup_memory_peak_gib == 3.0


def test_host_memory_snapshot_tolerates_malformed_cgroup_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup(
        monkeypatch,
        tmp_path,
        current="not-a-number",
        maximum="-1",
        peak="also-not-a-number",
    )
    _configure_psutil(monkeypatch)

    snapshot = host_memory._get_host_memory_snapshot()

    assert snapshot.process_rss_gib == 5.0
    assert snapshot.system_available_gib == 10.0
    assert snapshot.cgroup_memory_current_gib is None
    assert snapshot.cgroup_memory_max_gib is None
    assert snapshot.cgroup_memory_peak_gib is None


def test_host_memory_snapshot_tolerates_malformed_cgroup_membership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup(monkeypatch, tmp_path, use_mount_root=True)
    _configure_psutil(monkeypatch)
    (tmp_path / "proc-self-cgroup").write_text("not-a-unified-cgroup-entry\n")

    snapshot = host_memory._get_host_memory_snapshot()

    assert snapshot.process_rss_gib == 5.0
    assert snapshot.system_available_gib == 10.0
    assert snapshot.cgroup_memory_current_gib is None
    assert snapshot.cgroup_memory_max_gib is None
    assert snapshot.cgroup_memory_peak_gib is None


def test_host_memory_snapshot_preserves_cgroup_when_process_sampling_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup(monkeypatch, tmp_path)
    _configure_psutil(monkeypatch)
    monkeypatch.setattr(
        host_memory.psutil,
        "Process",
        lambda _pid=None: (_ for _ in ()).throw(RuntimeError("procfs unavailable")),
    )

    snapshot = host_memory._get_host_memory_snapshot()

    assert snapshot.process_rss_gib is None
    assert snapshot.system_available_gib == 10.0
    assert snapshot.cgroup_memory_current_gib == 2.0
    assert snapshot.cgroup_memory_max_gib == 4.0
    assert snapshot.cgroup_memory_peak_gib == 3.0


def test_host_memory_snapshot_preserves_rss_and_cgroup_when_system_sampling_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup(monkeypatch, tmp_path)
    _configure_psutil(monkeypatch)
    monkeypatch.setattr(
        host_memory.psutil,
        "virtual_memory",
        lambda: (_ for _ in ()).throw(RuntimeError("sysinfo unavailable")),
    )

    snapshot = host_memory._get_host_memory_snapshot()

    assert snapshot.process_rss_gib == 5.0
    assert snapshot.system_available_gib is None
    assert snapshot.cgroup_memory_current_gib == 2.0
    assert snapshot.cgroup_memory_max_gib == 4.0
    assert snapshot.cgroup_memory_peak_gib == 3.0


def test_host_memory_snapshot_preserves_psutil_when_cgroup_sampling_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_psutil(monkeypatch)
    monkeypatch.setattr(
        host_memory, "_PROC_SELF_CGROUP_PATH", tmp_path / "missing", raising=False
    )
    monkeypatch.setattr(
        host_memory, "_CGROUP_ROOT", tmp_path / "missing-cgroup", raising=False
    )

    snapshot = host_memory._get_host_memory_snapshot()

    assert snapshot.process_rss_gib == 5.0
    assert snapshot.system_available_gib == 10.0
    assert snapshot.cgroup_memory_current_gib is None
    assert snapshot.cgroup_memory_max_gib is None
    assert snapshot.cgroup_memory_peak_gib is None


def test_host_memory_event_formats_optional_values_and_deltas(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_cgroup(monkeypatch, tmp_path, maximum="max")
    _configure_psutil(monkeypatch, rss=7 * _GIB, available=8 * _GIB)
    before = host_memory.HostMemorySnapshot(5.0, None, 1.0, None, 2.0)

    snapshot = host_memory.emit_host_memory_event(
        event="test",
        phase="after",
        before_snapshot=before,
        include_deltas=True,
    )

    assert snapshot is not None
    assert capfd.readouterr().out.strip() == (
        "event=test phase=after process_rss_gib=7.000 "
        "process_rss_delta_gib=2.000 system_available_gib=8.000 "
        "system_available_delta_gib=unavailable cgroup_memory_current_gib=2.000 "
        "cgroup_memory_max_gib=unavailable cgroup_memory_peak_gib=3.000"
    )
