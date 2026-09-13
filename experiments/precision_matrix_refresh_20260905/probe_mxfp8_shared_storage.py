"""GPU diagnostic for DDP shared-storage restoration, not a production fix."""

import gc
import json
import os
import tempfile

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
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(
            128, 256, bias=False, device="cuda", dtype=torch.bfloat16
        )
        with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
            self.experts = GroupedLinear(
                2, 128, 256, bias=False, single_grouped_weight=True,
                params_dtype=torch.bfloat16, device="cuda",
            )
        for parameter in self.experts.parameters():
            parameter.allreduce = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dense(inputs) + self.experts(inputs, [32, 32])


def main() -> None:
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
            )
            module = MixedModule()
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
            for iteration in range(3):
                model.zero_grad_buffer()
                with fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
                    output = model(inputs)
                    loss = output.float().sum()
                loss.backward()
                torch.cuda.synchronize()
                actual_output = output.detach().cpu()
                gradients = {
                    name: parameter.main_grad.detach().cpu().clone()
                    for name, parameter in module.named_parameters()
                }
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
                for buffer in buffers:
                    buffer.offload_to_cpu(move_params=False, move_grads=True)
                    assert buffer.grad_data.untyped_storage().nbytes() == 0
                torch.cuda.empty_cache()
                # Occupy free allocator space so restoration cannot rely on old addresses.
                blocker = torch.empty(sum(sizes), device="cuda", dtype=torch.uint8)
                for buffer, size in zip(buffers, sizes):
                    buffer.reload_from_cpu(move_params=False, move_grads=True)
                    assert buffer.grad_data.untyped_storage().nbytes() == size
                torch.cuda.synchronize()
                del blocker
                print(json.dumps({"iteration": iteration, "shared_buffers": len(shared),
                                  "released_grad_bytes": sum(sizes), "exact_parity": True}),
                      flush=True)
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
