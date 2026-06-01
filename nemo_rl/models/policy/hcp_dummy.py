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

import torch


_DUMMY_ONE_LENGTH_TENSOR_KEYS = frozenset({"input_lengths", "expanded_lengths"})
_DUMMY_KNOWN_SEQUENCE_TENSOR_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_mask",
        "loss_mask",
        "mtp_loss_mask",
        "advantages",
        "generation_logprobs",
        "policy_logp",
        "prev_logprobs",
        "reference_policy_logprobs",
    }
)
_DUMMY_SEQUENCE_KEY_FRAGMENTS = (
    "token",
    "logp",
    "logprob",
    "log_probs",
    "mask",
    "position",
    "label",
    "reward",
)


def get_input_sequence_width(data: dict) -> int | None:
    input_ids = data.get("input_ids")
    if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
        return int(input_ids.shape[1])
    return None


def _trim_dummy_sequence_tensor(
    key: str, value: torch.Tensor, input_sequence_width: int | None
) -> torch.Tensor:
    if value.ndim < 2:
        return value

    is_known_sequence_tensor = key in _DUMMY_KNOWN_SEQUENCE_TENSOR_KEYS
    matches_input_width = (
        input_sequence_width is not None and int(value.shape[1]) == input_sequence_width
    )
    has_sequence_name = any(fragment in key for fragment in _DUMMY_SEQUENCE_KEY_FRAGMENTS)
    if not (
        is_known_sequence_tensor
        or matches_input_width
        or has_sequence_name
    ):
        return value

    if key == "attention_mask" and value.ndim > 2:
        slices = [slice(None)] * value.ndim
        for dim in range(1, value.ndim):
            if value.shape[dim] > 1 and (
                input_sequence_width is None
                or int(value.shape[dim]) == input_sequence_width
            ):
                slices[dim] = slice(0, 1)
        return value[tuple(slices)]

    return value[:, :1, ...]


def make_empty_hcp_dummy_tensor(
    key: str, value: torch.Tensor, input_sequence_width: int | None = None
) -> torch.Tensor:
    if key in _DUMMY_ONE_LENGTH_TENSOR_KEYS:
        return torch.ones_like(value)

    value = _trim_dummy_sequence_tensor(key, value, input_sequence_width)
    if key == "attention_mask":
        return torch.ones_like(value)
    return torch.zeros_like(value)
