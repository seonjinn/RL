"""Actual DDP/TE storage integration for the source-extracted worker swap."""

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import unwrap_model
from transformer_engine.pytorch.module import GroupedLinear

from test_reference_swap_lifecycle import TestReferenceSwapLifecycle


class TestDDPReferenceSwap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        cls.directory = tempfile.TemporaryDirectory()
        dist.init_process_group("nccl", init_method=f"file://{cls.directory.name}/init", rank=0, world_size=1)
        parallel_state.initialize_model_parallel()

    @classmethod
    def tearDownClass(cls) -> None:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
        cls.directory.cleanup()

    def test_wrapped_grouped_and_dense_snapshot(self) -> None:
        config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1,
                                   bf16=True, params_dtype=torch.bfloat16)
        module = torch.nn.Module()
        module.dense = torch.nn.Linear(128, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        module.experts = GroupedLinear(2, 128, 256, bias=False, single_grouped_weight=True,
                                      params_dtype=torch.bfloat16, device="cuda")
        self.assertIsNotNone(getattr(module.experts.weight, "rowwise_data", None))
        for param in module.experts.parameters():
            param.allreduce = False
        model = DistributedDataParallel(
            config, DistributedDataParallelConfig(use_distributed_optimizer=True),
            Float16Module(config, module),
        )
        self.assertTrue(model.buffers)
        self.assertTrue(model.expert_parallel_buffers)
        worker, swap = TestReferenceSwapLifecycle().make_worker()
        worker.model = model
        namespace = swap.__wrapped__.__globals__
        namespace.update(DistributedDataParallel=DistributedDataParallel, unwrap_model=unwrap_model)
        path = Path(__file__).parents[4] / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
        cls = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_apply_state_dict_to_model")
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        worker._apply_state_dict_to_model = MethodType(namespace[method.name], worker)
        worker.reference_state_dict = {}
        for name, value in model.state_dict().items():
            if isinstance(value, torch.Tensor) and "extra_state" not in name:
                worker.reference_state_dict[name] = torch.full(value.shape, 11, dtype=value.dtype)
            else:
                worker.reference_state_dict[name] = value
        buffers = [*model.buffers, *model.expert_parallel_buffers]
        pointers = None
        for value in (3, 7):
            with torch.no_grad():
                for buffer in buffers:
                    buffer.param_data.fill_(value)
            with self.assertRaisesRegex(ValueError, "injected forward failure"):
                with swap(worker):
                    for param in module.parameters():
                        live = getattr(param, "rowwise_data", param)
                        torch.testing.assert_close(live, torch.full_like(live, 11), rtol=0, atol=0)
                    raise ValueError("injected forward failure")
            for param in module.parameters():
                live = getattr(param, "rowwise_data", param)
                torch.testing.assert_close(live, torch.full_like(live, value), rtol=0, atol=0)
            current = [buffer.param_data_cpu.data_ptr() for buffer in buffers]
            if pointers is not None:
                self.assertEqual(current, pointers)
            pointers = current
            self.assertTrue(all(not buffer._cpu_snapshot_borrowed for buffer in buffers))


if __name__ == "__main__":
    unittest.main()
