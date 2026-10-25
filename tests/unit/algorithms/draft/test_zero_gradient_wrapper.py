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

from nemo_rl.algorithms.loss import ZeroGradientWrapper


def test_unused_parameters_receive_all_zero_gradients() -> None:
    """Catches a paused draft whose DDP grad buckets never see a gradient."""
    policy_parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    unused = [
        torch.nn.Parameter(torch.ones(3, 2)),
        torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16)),
    ]
    hook_calls: list[int] = []
    for index, parameter in enumerate(unused):
        parameter.register_hook(
            lambda grad, index=index: hook_calls.append(index) or grad
        )

    def loss_fn(scale: float) -> tuple[torch.Tensor, dict[str, float]]:
        return scale * policy_parameter.square().sum(), {"loss": 1.0}

    wrapper = ZeroGradientWrapper(loss_fn, unused)
    loss, metrics = wrapper(2.0)
    expected_loss, _ = loss_fn(2.0)
    loss.backward()

    assert torch.equal(loss, expected_loss)
    assert metrics == {"loss": 1.0}
    assert sorted(hook_calls) == [0, 1]
    for parameter in unused:
        assert parameter.grad is not None
        assert parameter.grad.dtype == parameter.dtype
        assert torch.count_nonzero(parameter.grad) == 0
    assert torch.equal(policy_parameter.grad, torch.tensor([8.0, 12.0]))


def test_wrapper_requires_parameters() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        ZeroGradientWrapper(lambda: (torch.zeros(()), {}), [])


@pytest.mark.parametrize(
    "parameter",
    [
        torch.nn.Parameter(torch.full((3,), float("nan"))),
        torch.nn.Parameter(torch.empty(0)),
    ],
    ids=["non_finite", "empty"],
)
def test_parameter_values_do_not_reach_the_loss(parameter: torch.nn.Parameter) -> None:
    """Catches a touch that reads draft values, which NaN or empty shards break."""
    policy_parameter = torch.nn.Parameter(torch.tensor([2.0]))
    wrapper = ZeroGradientWrapper(
        lambda: (policy_parameter.square().sum(), {}), [parameter]
    )

    loss, _ = wrapper()
    loss.backward()

    assert torch.equal(loss, torch.tensor(4.0))
    assert parameter.grad is not None
    assert parameter.grad.shape == parameter.shape
    assert torch.count_nonzero(parameter.grad) == 0
