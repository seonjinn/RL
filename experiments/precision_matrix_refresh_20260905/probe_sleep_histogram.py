"""Exercise sleep metadata reporting and byte preservation on a real CUDA pool."""

import ast
import contextlib
import io
import json
import os
import socket
import unittest
from pathlib import Path
from typing import Callable

import torch
from vllm.device_allocator.cumem import CuMemAllocator


def load_report() -> Callable:
    path = Path(__file__).parents[2] / "nemo_rl/models/generation/vllm/vllm_backend.py"
    tree = ast.parse(path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "report_sleep_memory"
    )
    namespace = {"torch": torch, "os": os, "socket": socket}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["report_sleep_memory"]


class SleepHistogramProbe(unittest.TestCase):
    def test_real_pool_roundtrips(self) -> None:
        torch.cuda.set_device(0)
        allocator = CuMemAllocator.get_instance()
        report = load_report()
        with allocator.use_memory_pool(tag="histogram_probe"):
            values = [
                torch.empty(n, dtype=torch.uint8, device="cuda")
                for n in (3 * 1024**2, 7 * 1024**2, 19 * 1024**2)
            ]

        def observe(phase: str) -> dict:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report(None, phase)
            lines = [
                line
                for line in output.getvalue().splitlines()
                if line.startswith("NRL_SLEEP_MEMORY ")
            ]
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0].split(" ", 1)[1])
            histogram = record["handle_size_histogram"]["histogram_probe"]
            counts = record["tags"]["histogram_probe"]
            self.assertEqual(sum(histogram.values()), counts["handles"])
            self.assertEqual(
                sum(int(size) * count for size, count in histogram.items()),
                counts["capacity"],
            )
            print(lines[0], flush=True)
            return record

        for cycle in range(3):
            for index, tensor in enumerate(values):
                tensor.fill_(17 + cycle * 31 + index)
            torch.cuda.synchronize()
            before = observe("pre_sleep")
            allocator.sleep(offload_tags=("histogram_probe",))
            asleep = observe("post_sleep")
            self.assertEqual(
                before["handle_size_histogram"], asleep["handle_size_histogram"]
            )
            self.assertEqual(
                asleep["backup_storage"]["union_bytes"],
                asleep["tags"]["histogram_probe"]["capacity"],
            )
            allocator.wake_up()
            for index, tensor in enumerate(values):
                self.assertTrue(torch.all(tensor == 17 + cycle * 31 + index).item())
            awake = observe("post_wake")
            self.assertEqual(awake["backup_storage"]["union_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
