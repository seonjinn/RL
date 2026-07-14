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

from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.mcore


def test_full_graph_forward_step_preserves_a2a_schedule_plan() -> None:
    from megatron.core.packed_seq_params import PackedSeqParams

    from nemo_rl.algorithms.loss import NLLLossFn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron.data import ProcessedMicrobatch
    from nemo_rl.models.megatron.full_cuda_graph import (
        FULL_CUDA_GRAPH_GLOBAL_VALID_SEQS,
        FULL_CUDA_GRAPH_GLOBAL_VALID_TOKS,
        ProcessedMicrobatchStaticBufferLoader,
    )
    from nemo_rl.models.megatron.train import (
        LossPostProcessor,
        megatron_forward_backward,
    )

    observed_valid_seqs: Optional[torch.Tensor] = None
    observed_valid_toks: Optional[torch.Tensor] = None
    static_microbatch: Optional[ProcessedMicrobatch] = None

    class RecordingLossPostProcessor(LossPostProcessor):
        def __call__(
            self,
            data_dict: BatchedDataDict[Any],
            packed_seq_params: Optional[PackedSeqParams] = None,
            global_valid_seqs: Optional[torch.Tensor] = None,
            global_valid_toks: Optional[torch.Tensor] = None,
        ) -> Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, Any]]]:
            nonlocal observed_valid_seqs, observed_valid_toks
            observed_valid_seqs = global_valid_seqs
            observed_valid_toks = global_valid_toks
            return lambda output_tensor: (output_tensor.new_zeros(()), {})

    input_ids = torch.tensor([[1, 2, 3]])
    microbatch = ProcessedMicrobatch(
        data_dict=BatchedDataDict(
            {
                "input_ids": input_ids,
                "token_mask": torch.ones_like(input_ids),
                "sample_mask": torch.ones(1),
            }
        ),
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids,
        attention_mask=torch.ones(1, 3),
        position_ids=torch.tensor([[0, 1, 2]]),
        packed_seq_params=None,
        cu_seqlens_padded=None,
    )
    model = MagicMock()
    schedule_plan = MagicMock()
    model.build_schedule_plan.return_value = schedule_plan
    static_loader = ProcessedMicrobatchStaticBufferLoader()

    def fake_raw_schedule(
        *,
        forward_step_func: Callable[..., tuple[Any, Callable[..., Any]]],
        data_iterator: Any,
        model: Any,
        **_: Any,
    ) -> Any:
        nonlocal static_microbatch
        attached_microbatch = next(data_iterator)
        static_microbatch = static_loader(attached_microbatch, "training", 0)
        output, _ = forward_step_func(
            iter([static_microbatch]),
            model,
            return_schedule_plan=True,
        )
        return output

    output = megatron_forward_backward(
        model=model,
        data_iterator=iter([microbatch]),
        num_microbatches=1,
        seq_length=3,
        mbs=1,
        post_processing_fn=RecordingLossPostProcessor(
            loss_fn=NLLLossFn(),
            cfg={"sequence_packing": {"enabled": False}},
        ),
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(2.0),
        forward_backward_func=fake_raw_schedule,
    )

    assert static_microbatch is not None
    assert output is schedule_plan
    model.build_schedule_plan.assert_called_once_with(
        input_ids=static_microbatch.input_ids_cp_sharded,
        position_ids=static_microbatch.position_ids,
        attention_mask=static_microbatch.attention_mask,
    )
    assert (
        observed_valid_seqs
        is static_microbatch.data_dict[FULL_CUDA_GRAPH_GLOBAL_VALID_SEQS]
    )
    assert (
        observed_valid_toks
        is static_microbatch.data_dict[FULL_CUDA_GRAPH_GLOBAL_VALID_TOKS]
    )
