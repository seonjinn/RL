"""Execute the worker swap methods without importing the full Ray worker stack."""

import ast
import gc
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

from nemo_rl.models.policy.reference_snapshot import borrowed_cpu_parameter_views


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([3.0]))


class TestReferenceSwapLifecycle(unittest.TestCase):
    def make_worker(self):
        path = Path(__file__).parents[4] / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "MegatronPolicyWorkerImpl")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "use_reference_model")
        namespace = {
            "contextmanager": contextmanager, "ExitStack": ExitStack,
            "torch": torch, "gc": gc, "DistributedDataParallel": type("UnusedDDP", (), {}),
            "unwrap_model": lambda model: model,
            "borrowed_cpu_parameter_views": borrowed_cpu_parameter_views,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        worker = SimpleNamespace(
            model=Model(), cfg={"megatron_cfg": {"empty_unused_memory_level": 0}},
            reference_state_dict={"weight": torch.tensor([11.0])},
            should_disable_forward_pre_hook=False, sampling_params=None,
            _log_host_storage=lambda *args: None,
        )

        def apply(state, **kwargs):
            worker.model.state_dict()["weight"].copy_(state["weight"])

        worker._apply_state_dict_to_model = apply
        return worker, namespace["use_reference_model"]

    def test_forward_exception_restores_policy(self) -> None:
        worker, swap = self.make_worker()
        with self.assertRaisesRegex(ValueError, "forward failed"):
            with swap(worker):
                self.assertEqual(worker.model.weight.item(), 11.0)
                raise ValueError("forward failed")
        self.assertEqual(worker.model.weight.item(), 3.0)

    def test_partial_reference_apply_restores_policy(self) -> None:
        worker, swap = self.make_worker()
        apply = worker._apply_state_dict_to_model

        def failing_apply(state, **kwargs):
            apply(state, **kwargs)
            if state is worker.reference_state_dict:
                raise ValueError("partial reference apply")

        worker._apply_state_dict_to_model = failing_apply
        with self.assertRaisesRegex(ValueError, "partial reference apply"):
            with swap(worker):
                self.fail("must not run a partially loaded reference")
        self.assertEqual(worker.model.weight.item(), 3.0)

    def test_normal_repeated_swap(self) -> None:
        worker, swap = self.make_worker()
        for _ in range(2):
            with swap(worker):
                self.assertEqual(worker.model.weight.item(), 11.0)
            self.assertEqual(worker.model.weight.item(), 3.0)

    def test_borrowed_restore_precedes_release(self) -> None:
        worker, swap = self.make_worker()
        active = []
        backup = torch.empty_like(worker.model.weight)

        @contextmanager
        def borrow():
            backup.copy_(worker.model.weight.detach())
            active.append(True)
            try:
                yield {worker.model.weight: backup}
            finally:
                active.pop()

        swap.__wrapped__.__globals__["DistributedDataParallel"] = Model
        worker.model.buffers = [SimpleNamespace(borrow_cpu_param_snapshot=borrow)]
        worker.model.expert_parallel_buffers = []
        apply = worker._apply_state_dict_to_model

        def checked_apply(state, **kwargs):
            self.assertTrue(active)
            if state is not worker.reference_state_dict:
                self.assertEqual(state["weight"].data_ptr(), backup.data_ptr())
            apply(state, **kwargs)

        worker._apply_state_dict_to_model = checked_apply
        with self.assertRaisesRegex(ValueError, "reference failure"):
            with swap(worker):
                raise ValueError("reference failure")
        self.assertFalse(active)
        self.assertEqual(worker.model.weight.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
