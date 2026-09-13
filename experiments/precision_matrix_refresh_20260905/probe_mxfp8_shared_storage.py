"""GPU diagnostic for DDP shared-storage restoration, not a production fix."""

import argparse
import ast
import gc
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import fp8_autocast, fp8_model_init
from transformer_engine.pytorch.module import GroupedLinear


class MixedModule(torch.nn.Module):
    def __init__(self, fused_wgrad: bool) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(
            128, 256, bias=False, device="cuda", dtype=torch.bfloat16
        )
        with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
            self.experts = GroupedLinear(
                2, 128, 256, bias=False, single_grouped_weight=True,
                params_dtype=torch.bfloat16, device="cuda",
                fuse_wgrad_accumulation=fused_wgrad,
            )
        for parameter in self.experts.parameters():
            parameter.allreduce = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dense(inputs) + self.experts(inputs, [32, 32])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-wgrad", action="store_true")
    parser.add_argument("--preserve-shared-storage", action="store_true")
    parser.add_argument("--trace-gradients", action="store_true")
    args = parser.parse_args()
    move_model = None
    if args.preserve_shared_storage:
        path = Path(__file__).parents[2] / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
        cls = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "move_model")
        namespace = {"torch": torch, "DistributedDataParallel": DistributedDataParallel}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        move_model = namespace["move_model"]
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    torch.manual_seed(123)
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group(
            "nccl", init_method=f"file://{directory}/init", rank=0, world_size=1
        )
        parallel_state.initialize_model_parallel()
        try:
            config = TransformerConfig(
                num_layers=1, hidden_size=128, num_attention_heads=1,
                bf16=True, params_dtype=torch.bfloat16,
                fp8="e4m3", fp8_recipe="mxfp8",
                gradient_accumulation_fusion=args.fused_wgrad,
            )
            module = MixedModule(args.fused_wgrad)
            model = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True,
                    fp8_param_gather=True,
                    reuse_grad_buf_for_mxfp8_param_ag=True,
                    overlap_param_gather=False,
                ),
                Float16Module(config, module),
            )
            buffers = [*model.buffers, *model.expert_parallel_buffers]
            shared = [buffer for buffer in buffers if hasattr(buffer, "shared_buffer")]
            assert shared, "Probe must exercise real MXFP8 shared storage"
            inputs = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
            expected: dict[str, torch.Tensor] = {}
            expected_output = None
            hook_handles = []
            if args.trace_gradients:
                def trace_gradient(name: str):
                    def hook(gradient: torch.Tensor) -> None:
                        print(json.dumps({"autograd_parameter": name,
                                          "gradient_type": type(gradient).__name__,
                                          "finite": bool(torch.isfinite(gradient).all()),
                                          "nonzero": int(torch.count_nonzero(gradient))}),
                              flush=True)
                    return hook
                for name, parameter in module.named_parameters():
                    hook_handles.append(parameter.register_hook(trace_gradient(name)))
            for iteration in range(3):
                model.zero_grad_buffer()
                with fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
                    output = model(inputs)
                    loss = output.float().sum()
                loss.backward()
                torch.cuda.synchronize()
                if args.trace_gradients:
                    for name, parameter in module.named_parameters():
                        print(json.dumps({"iteration": iteration, "parameter": name,
                                          "type": type(parameter).__name__,
                                          "requires_grad": parameter.requires_grad,
                                          "grad_is_none": parameter.grad is None,
                                          "grad_added_to_main_grad": parameter.grad_added_to_main_grad,
                                          "main_grad_nonzero": int(torch.count_nonzero(parameter.main_grad))}),
                              flush=True)
                actual_output = output.detach().cpu()
                gradients = {
                    name: parameter.main_grad.detach().cpu().clone()
                    for name, parameter in module.named_parameters()
                }
                for name, gradient in gradients.items():
                    finite = bool(torch.isfinite(gradient).all())
                    nonzero = int(torch.count_nonzero(gradient))
                    assert finite and nonzero > 0, (
                        f"iteration={iteration} parameter={name} finite={finite} nonzero={nonzero}"
                    )
                if iteration == 0:
                    expected = gradients
                    expected_output = actual_output
                else:
                    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
                    for name, gradient in gradients.items():
                        torch.testing.assert_close(gradient, expected[name], rtol=0, atol=0)
                del output, loss
                gc.collect()
                sizes = [buffer.grad_data.untyped_storage().nbytes() for buffer in buffers]
                if move_model is not None:
                    shared_states = [
                        (buffer.grad_data.untyped_storage()._cdata, buffer.grad_data.clone())
                        for buffer in shared
                    ]
                    for _ in range(2):
                        move_model(SimpleNamespace(), model, "cpu", preserve_shared_param_grad=True)
                    for buffer, (identity, values) in zip(shared, shared_states):
                        assert buffer.grad_data.untyped_storage()._cdata == identity
                        torch.testing.assert_close(buffer.grad_data, values, rtol=0, atol=0)
                    del shared_states, values
                    for buffer, size in zip(buffers, sizes):
                        expected_size = size if any(buffer is item for item in shared) else 0
                        assert buffer.grad_data.untyped_storage().nbytes() == expected_size
                else:
                    for buffer in buffers:
                        buffer.offload_to_cpu(move_params=False, move_grads=True)
                        assert buffer.grad_data.untyped_storage().nbytes() == 0
                released = sum(sizes) - sum(
                    buffer.grad_data.untyped_storage().nbytes() for buffer in buffers
                )
                assert released > 0, "At least one independent gradient buffer must be freed"
                torch.cuda.empty_cache()
                # Occupy free allocator space so restoration cannot rely on old addresses.
                blocker = torch.empty(sum(sizes), device="cuda", dtype=torch.uint8)
                if move_model is not None:
                    move_model(SimpleNamespace(), model, "cuda")
                for buffer, size in zip(buffers, sizes):
                    if move_model is None:
                        buffer.reload_from_cpu(move_params=False, move_grads=True)
                    assert buffer.grad_data.untyped_storage().nbytes() == size
                torch.cuda.synchronize()
                del blocker
                print(json.dumps({"iteration": iteration, "shared_buffers": len(shared),
                                  "fused_wgrad": args.fused_wgrad,
                                  "preserve_shared_storage": args.preserve_shared_storage,
                                  "released_grad_bytes": released, "exact_parity": True}),
                      flush=True)
            for handle in hook_handles:
                handle.remove()
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
