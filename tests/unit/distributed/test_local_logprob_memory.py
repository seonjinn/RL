# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from nemo_rl.distributed.model_utils import get_next_token_logprobs_from_logits


def test_local_next_token_logprobs_follow_logits_device():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    input_ids = torch.tensor([[0, 1, 0]])

    actual = get_next_token_logprobs_from_logits(input_ids, logits)
    expected = (
        torch.log_softmax(logits[:, :-1], dim=-1)
        .gather(-1, input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_local_logprobs_match_full_forward_and_backward(dtype: torch.dtype):
    torch.manual_seed(17)
    values = torch.randn(2, 8, 13).to(dtype)
    input_ids = torch.tensor([[0, 2, 4, 6, 8, 10, 12, 1], [1, 3, 5, 7, 9, 11, 0, 2]])
    weights = torch.randn(2, 7)
    reference_logits = values.clone().requires_grad_(True)
    chunked_logits = values.clone().requires_grad_(True)

    reference = torch.log_softmax(reference_logits.float()[:, :-1], dim=-1)
    reference = reference.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    actual = get_next_token_logprobs_from_logits(
        input_ids, chunked_logits, chunk_size=2
    )

    (reference * weights).sum().backward()
    (actual * weights).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        chunked_logits.grad.float(),
        reference_logits.grad.float(),
        rtol=3e-2 if dtype == torch.bfloat16 else 1e-5,
        atol=1e-2 if dtype == torch.bfloat16 else 1e-5,
    )


def test_chunked_local_logprobs_do_not_save_full_fp32_vocabulary_tensor():
    torch.manual_seed(19)
    logits = torch.randn(1, 7, 17, dtype=torch.bfloat16, requires_grad=True)
    input_ids = torch.tensor([[0, 2, 4, 6, 8, 10, 12]])
    saved_tensors: list[tuple[tuple[int, ...], torch.dtype]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_tensors.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = get_next_token_logprobs_from_logits(input_ids, logits, chunk_size=2)
        result.sum().backward()

    assert logits.grad is not None
    assert not any(
        dtype == torch.float32 and shape[-1] == 17 and len(shape) == 3
        for shape, dtype in saved_tensors
    )
