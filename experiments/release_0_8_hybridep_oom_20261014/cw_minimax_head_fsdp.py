"""Check repeated chunked LM-head calls through an FSDP2 wrapper on one H100."""

import os

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard

from nemo_rl.models.automodel.train import chunked_lm_head_logprobs


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    head = torch.nn.Linear(32, 97, bias=False, device="cuda")
    reference_head = torch.nn.Linear(32, 97, bias=False, device="cuda")
    reference_head.weight.data.copy_(head.weight.data)
    fully_shard(head)
    tokens = torch.randint(0, 97, (2, 17), device="cuda")
    hidden = torch.randn(2, 17, 32, device="cuda", requires_grad=True)
    actual = chunked_lm_head_logprobs(hidden, tokens, head, 4)
    actual.sum().backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = head.weight.grad.to_local().detach().clone()

    reference_hidden = hidden.detach().clone().requires_grad_()
    reference = (
        torch.log_softmax(reference_head(reference_hidden[:, :-1]).float(), dim=-1)
        .gather(-1, tokens[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )
    reference.sum().backward()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_hidden_grad, reference_hidden.grad)
    torch.testing.assert_close(actual_weight_grad, reference_head.weight.grad)
    print("MiniMax chunked LM-head FSDP2 forward/backward passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
