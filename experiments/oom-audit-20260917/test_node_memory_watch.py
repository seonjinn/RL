import tempfile
import unittest
from pathlib import Path

from node_memory_watch import optional_read, read_cgroup_chain


class TestCgroupChain(unittest.TestCase):
    def test_missing_pressure_file_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                optional_read(Path(directory) / "missing-pressure"),
                {"error": "FileNotFoundError"},
            )

    def test_reads_ancestors_and_stops_at_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "job" / "step"
            leaf.mkdir(parents=True)
            (root / "job" / "memory.events").write_text("oom_kill 1\n")
            (leaf / "memory.current").write_text("123\n")
            records = read_cgroup_chain(leaf, root)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["memory.current"], "123")
            self.assertEqual(records[1]["memory.events"], "oom_kill 1")

    def test_rejects_path_outside_mount(self) -> None:
        with self.assertRaises(ValueError):
            read_cgroup_chain(Path("/outside"), Path("/sys/fs/cgroup"))
