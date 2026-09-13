"""GB200 probe of the actual optimizer movement method, without model loading."""

import ast
import contextlib
import io
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class ChainedSentinel:
    pass


def load_move_optimizer():
    path = Path(__file__).parents[2] / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "move_optimizer")
    namespace = {"torch": torch, "os": os, "time": time, "ChainedOptimizer": ChainedSentinel}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["move_optimizer"]


class OptimizerCopyProbe(unittest.TestCase):
    def test_roundtrip_and_split_diagnostic(self) -> None:
        move = load_move_optimizer()
        for diagnostic in (False, True):
            os.environ["NRL_HOST_STORAGE_DIAGNOSTICS"] = str(int(diagnostic))
            for dtype in (torch.float32, torch.bfloat16):
                base = torch.arange(256, device="cuda", dtype=dtype).reshape(16, 16)
                for value in (base, base.T, base[::2, ::2], base[0, 0]):
                    with self.subTest(diagnostic=diagnostic, dtype=dtype, shape=value.shape):
                        expected = value.cpu()
                        cpu_state = torch.tensor(7)
                        states = {0: {"exp_avg": value, "step": cpu_state, "metadata": 3}}
                        worker = SimpleNamespace(optimizer=SimpleNamespace(_get_state=lambda: states))
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            for _ in range(3):
                                move(worker, "cpu")
                                actual = states[0]["exp_avg"]
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                                self.assertEqual(actual.stride(), expected.stride())
                                self.assertEqual(actual.dtype, expected.dtype)
                                self.assertEqual(states[0]["metadata"], 3)
                                move(worker, "cuda")
                        if diagnostic:
                            self.assertIn("allocation_s=", output.getvalue())
                            self.assertIn("copy_s=", output.getvalue())
                        else:
                            self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
