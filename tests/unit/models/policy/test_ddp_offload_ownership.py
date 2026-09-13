"""Source-extracted offload dispatch tests; full GPU lifecycle is separate."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class Buffer:
    def __init__(self, param: torch.Tensor, grad: torch.Tensor) -> None:
        self.param_data = param
        self.grad_data = grad
        self.calls: list[dict[str, bool]] = []

    def offload_to_cpu(self, **kwargs: bool) -> None:
        self.calls.append(kwargs)


class DDP:
    def __init__(self, buffers: list[Buffer]) -> None:
        self.buffers = buffers[:1]
        self.expert_parallel_buffers = buffers[1:]


def move_model_method():
    path = Path(__file__).parents[4] / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "move_model")
    namespace = {
        "torch": torch, "DistributedDataParallel": DDP,
        "FullyShardedDataParallelV1": type("FSDP1", (), {}),
        "FullyShardedDataParallelV2": type("FSDP2", (), {}),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["move_model"]


class TestDDPOffloadOwnership(unittest.TestCase):
    def test_preserves_aliases_but_offloads_independent_storage(self) -> None:
        backing = torch.empty(32, dtype=torch.float32)
        independent = Buffer(torch.empty(16), torch.empty(16))
        shared = Buffer(backing[4:12].view(torch.bfloat16), backing)
        model = DDP([independent, shared])
        move_model_method()(SimpleNamespace(), model, "cpu", preserve_shared_param_grad=True)
        self.assertEqual(independent.calls, [{"move_params": True, "move_grads": True}])
        self.assertEqual(shared.calls, [{"move_params": False, "move_grads": False}])

    def test_empty_independent_storages_are_not_aliases(self) -> None:
        buffer = Buffer(torch.empty(0), torch.empty(0))
        move_model_method()(SimpleNamespace(), DDP([buffer]), "cpu",
                            preserve_shared_param_grad=True)
        self.assertEqual(buffer.calls, [{"move_params": True, "move_grads": True}])

    def test_preserves_requested_parameter_residency(self) -> None:
        buffer = Buffer(torch.empty(16), torch.empty(16))
        move_model_method()(SimpleNamespace(), DDP([buffer]), "cpu", move_params=False,
                            preserve_shared_param_grad=True)
        self.assertEqual(buffer.calls, [{"move_params": False, "move_grads": True}])

    def test_non_ddp_retains_conservative_behavior(self) -> None:
        class Module:
            def to(self, **kwargs: object) -> None:
                raise AssertionError("Shared storage ownership is unknown outside ordinary DDP")

        move_model_method()(SimpleNamespace(), Module(), "cpu", preserve_shared_param_grad=True)


if __name__ == "__main__":
    unittest.main()
