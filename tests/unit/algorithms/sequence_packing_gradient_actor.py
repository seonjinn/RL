# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Ray actor for sequence packing gradient tests.

Separated from test_sequence_packing_gradients.py to avoid importing pytest
in Ray worker environments that use PY_EXECUTABLES.MCORE.
"""

import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import ray
import torch

from nemo_rl.algorithms.loss import (
    ClippedPGLossConfig,
    ClippedPGLossFn,
    SequencePackingLossWrapper,
)
from nemo_rl.algorithms.loss.utils import prepare_loss_input
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@dataclass(frozen=True)
class _PolicyGradientOracleResult:
    policy_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    valid_logits: torch.Tensor
    valid_input_grad: torch.Tensor
    padding_input_grad: torch.Tensor
    router_grad: torch.Tensor
    expert_grads: dict[str, torch.Tensor | None]


def _assert_fixed_capacity_packed_ownership(
    *,
    structural_padding_mask: torch.Tensor,
    sample_ids: torch.Tensor,
    num_samples: torch.Tensor,
    max_samples: int,
) -> None:
    """Assert the literal CP1 ownership geometry used by the policy oracle."""
    expected_padding_mask = torch.tensor(
        [
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ]
        ],
        dtype=torch.bool,
        device=structural_padding_mask.device,
    )
    expected_sample_ids = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        dtype=torch.int64,
        device=sample_ids.device,
    )
    torch.testing.assert_close(structural_padding_mask, expected_padding_mask)
    torch.testing.assert_close(sample_ids, expected_sample_ids)
    assert num_samples.ndim == 0
    assert int(num_samples.item()) == 2
    assert max_samples == 3


@ray.remote(num_gpus=1)
class SequencePackingGradientTestActor:
    def __init__(self, cp_size):
        self.cp_size = cp_size
        self.env_vars = dict(os.environ)

    def _test_packed_policy_gradient_ownership(self) -> None:
        # These imports require the MCore Ray worker environment and must not run
        # when this actor module is collected by the non-MCore pytest driver.
        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_submodules,
        )
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
        from megatron.core.transformer.moe.moe_logging import (
            destroy_moe_metrics_tracker,
            get_moe_metrics_tracker,
        )
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
        from megatron.core.transformer.spec_utils import get_submodules
        from megatron.core.transformer.transformer_config import TransformerConfig

        from nemo_rl.models.megatron.data import (
            _pack_sequences_for_megatron_with_geometry,
        )

        hidden_size = 16
        vocab_size = 11
        num_experts = 4
        expert_parameter_names = tuple(
            f"experts.local_experts.{expert_index}.linear_fc{projection}.weight"
            for expert_index in range(num_experts)
            for projection in (1, 2)
        )

        parallel_state.destroy_model_parallel()
        destroy_moe_metrics_tracker()
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                expert_tensor_parallel_size=1,
                create_gloo_process_groups=False,
            )
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=[
                    "tp",
                    "cp",
                    "tp_cp",
                    "tp_dp_cp",
                    "ep",
                    "expt_tp",
                    "tp_ep",
                    "expt_dp",
                ]
            )
            config = TransformerConfig(
                num_layers=1,
                hidden_size=hidden_size,
                num_attention_heads=4,
                ffn_hidden_size=32,
                moe_ffn_hidden_size=32,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
                expert_tensor_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                num_moe_experts=num_experts,
                moe_router_topk=2,
                moe_router_load_balancing_type="seq_aux_loss",
                moe_aux_loss_coeff=0.1,
                moe_router_score_function="softmax",
                moe_router_dtype="fp32",
                moe_token_dispatcher_type="allgather",
                moe_expert_capacity_factor=None,
                moe_pad_expert_input_to_capacity=False,
                moe_router_fusion=False,
                moe_permute_fusion=False,
                moe_grouped_gemm=False,
                use_transformer_engine_op_fuser=False,
                add_bias_linear=False,
                calculate_per_token_loss=False,
                use_cpu_initialization=True,
                params_dtype=torch.float32,
                fp16=False,
                bf16=False,
                transformer_impl="local",
            )

            def build_moe_layer() -> MoELayer:
                mlp_spec = get_gpt_layer_local_submodules(
                    num_experts=num_experts,
                    moe_grouped_gemm=False,
                ).mlp
                submodules = get_submodules(mlp_spec)
                assert isinstance(submodules, MoESubmodules)
                layer = MoELayer(
                    config,
                    submodules,
                    layer_number=1,
                    pg_collection=pg_collection,
                ).cuda()
                layer.set_layer_number(1)
                return layer.train()

            torch.manual_seed(1729)
            torch.cuda.manual_seed_all(1729)
            padded_layer = build_moe_layer()
            with torch.no_grad():
                padded_layer.router.weight.zero_()
                padded_layer.router.weight[:, :num_experts].copy_(
                    torch.eye(
                        num_experts,
                        device=padded_layer.router.weight.device,
                        dtype=padded_layer.router.weight.dtype,
                    )
                )
            packed_layer = build_moe_layer()
            packed_layer.load_state_dict(
                {
                    name: value.detach().clone()
                    for name, value in padded_layer.state_dict().items()
                }
            )

            padded_head = torch.nn.Linear(
                hidden_size,
                vocab_size,
                bias=False,
                device="cuda",
                dtype=torch.float32,
            )
            packed_head = torch.nn.Linear(
                hidden_size,
                vocab_size,
                bias=False,
                device="cuda",
                dtype=torch.float32,
            )
            packed_head.load_state_dict(
                {
                    name: value.detach().clone()
                    for name, value in padded_head.state_dict().items()
                }
            )

            input_ids = torch.tensor(
                [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]],
                dtype=torch.int64,
                device="cuda",
            )
            seq_lengths = torch.tensor([3, 5], dtype=torch.int64, device="cuda")
            packed_output = _pack_sequences_for_megatron_with_geometry(
                input_ids=input_ids,
                seq_lengths=seq_lengths,
                pad_individual_seqs_to_multiple_of=4,
                pad_packed_seq_to_multiple_of=1,
                pad_packed_seq_to=16,
                cp_rank=0,
                cp_size=1,
                thd_max_packed_sequences=4,
            )
            packed_seq_params = packed_output.packed_seq_params
            assert packed_seq_params.seq_aux_loss_sample_ids is not None
            assert packed_seq_params.seq_aux_loss_num_samples is not None
            assert packed_seq_params.seq_aux_loss_max_samples is not None
            _assert_fixed_capacity_packed_ownership(
                structural_padding_mask=packed_output.structural_padding_mask,
                sample_ids=packed_seq_params.seq_aux_loss_sample_ids,
                num_samples=packed_seq_params.seq_aux_loss_num_samples,
                max_samples=packed_seq_params.seq_aux_loss_max_samples,
            )
            torch.testing.assert_close(
                packed_output.cu_seqlens_padded,
                torch.tensor([0, 4, 12], dtype=torch.int32, device="cuda"),
            )
            assert packed_output.packed_geometry.logical_tokens == 8
            assert packed_output.packed_geometry.padded_tokens == 12
            assert packed_output.packed_geometry.capacity_tokens == 16
            assert packed_output.packed_geometry.real_sequence_count == 2

            routing_patterns = torch.tensor(
                [
                    [4.0, 3.0, 2.0, 1.0],
                    [1.0, 4.0, 3.0, 2.0],
                    [2.0, 1.0, 4.0, 3.0],
                    [3.0, 2.0, 1.0, 4.0],
                ],
                device="cuda",
            )
            valid_hidden = torch.zeros(
                8, hidden_size, dtype=torch.float32, device="cuda"
            )
            valid_hidden[:, :num_experts] = routing_patterns.repeat(2, 1)

            padded_hidden = torch.zeros(
                5, 2, hidden_size, dtype=torch.float32, device="cuda"
            )
            padded_hidden[:3, 0] = valid_hidden[:3]
            padded_hidden[:5, 1] = valid_hidden[3:]
            padded_padding_mask = torch.tensor(
                [
                    [False, False, False, True, True],
                    [False, False, False, False, False],
                ],
                dtype=torch.bool,
                device="cuda",
            )

            packed_hidden = torch.zeros(
                16, 1, hidden_size, dtype=torch.float32, device="cuda"
            )
            physical_starts = packed_output.cu_seqlens_padded[:-1].tolist()
            packed_hidden[physical_starts[0] : physical_starts[0] + 3, 0] = (
                valid_hidden[:3]
            )
            packed_hidden[physical_starts[1] : physical_starts[1] + 5, 0] = (
                valid_hidden[3:]
            )
            packed_padding_mask = packed_output.structural_padding_mask

            token_mask = (~padded_padding_mask).to(dtype=torch.float32)
            advantages = torch.tensor(
                [
                    [0.0, 0.25, -0.5, 0.0, 0.0],
                    [0.0, -0.75, 0.5, 1.0, -0.25],
                ],
                dtype=torch.float32,
                device="cuda",
            )
            logprob_zeros = torch.zeros_like(advantages)
            loss_config = ClippedPGLossConfig(
                force_on_policy_ratio=True,
                reference_policy_kl_penalty=0.0,
                ratio_clip_c=None,
            )
            global_valid_seqs = torch.tensor(2.0, device="cuda")
            global_valid_toks = token_mask[:, 1:].sum()
            torch.testing.assert_close(
                global_valid_toks,
                torch.tensor(6.0, device="cuda"),
            )

            def run_case(
                *,
                layer: MoELayer,
                head: torch.nn.Linear,
                hidden: torch.Tensor,
                padding_mask: torch.Tensor,
                packed_params: PackedSeqParams | None,
                packed: bool,
            ) -> _PolicyGradientOracleResult:
                destroy_moe_metrics_tracker()
                layer.zero_grad(set_to_none=True)
                head.zero_grad(set_to_none=True)
                model_hidden = hidden.detach().clone().requires_grad_(True)
                model_output, mlp_bias = layer(
                    model_hidden,
                    padding_mask=padding_mask,
                    packed_seq_params=packed_params,
                )
                assert mlp_bias is None
                logits = head(model_output)
                if packed:
                    valid_logits = logits[:, 0][~padding_mask[0]]
                else:
                    valid_logits = logits.transpose(0, 1)[~padding_mask]
                assert valid_logits.shape == (8, vocab_size)

                policy_logits = logits.new_zeros((2, 5, vocab_size))
                policy_logits[~padded_padding_mask] = valid_logits
                data = BatchedDataDict(
                    {
                        "input_ids": input_ids.clone(),
                        "token_mask": token_mask.clone(),
                        "sample_mask": torch.ones(2, device="cuda"),
                        "advantages": advantages.clone(),
                        "prev_logprobs": logprob_zeros.clone(),
                        "generation_logprobs": logprob_zeros.clone(),
                        "reference_policy_logprobs": logprob_zeros.clone(),
                    }
                )
                loss_fn = ClippedPGLossFn(loss_config)
                loss_input, data = prepare_loss_input(policy_logits, data, loss_fn)
                policy_loss, _ = loss_fn(
                    data=data,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                    **loss_input,
                )
                tracker = get_moe_metrics_tracker()
                auxiliary_loss = (
                    tracker.metrics["seq_load_balancing_loss"]
                    .values[0]
                    .detach()
                    .clone()
                    * 0.1
                )
                MoEAuxLossAutoScaler.set_loss_scale(
                    torch.ones((), dtype=torch.float32, device="cuda")
                )
                policy_loss.backward()

                assert model_hidden.grad is not None
                if packed:
                    valid_input_grad = model_hidden.grad[:, 0][~padding_mask[0]]
                    padding_input_grad = model_hidden.grad[:, 0][padding_mask[0]]
                else:
                    sample_major_grad = model_hidden.grad.transpose(0, 1)
                    valid_input_grad = sample_major_grad[~padding_mask]
                    padding_input_grad = sample_major_grad[padding_mask]

                parameters = dict(layer.named_parameters())
                router_grad = parameters["router.weight"].grad
                assert router_grad is not None
                expert_grads = {
                    name: (
                        None
                        if parameters[name].grad is None
                        else parameters[name].grad.detach().clone()
                    )
                    for name in expert_parameter_names
                }
                return _PolicyGradientOracleResult(
                    policy_loss=policy_loss.detach().clone(),
                    auxiliary_loss=auxiliary_loss,
                    valid_logits=valid_logits.detach().clone(),
                    valid_input_grad=valid_input_grad.detach().clone(),
                    padding_input_grad=padding_input_grad.detach().clone(),
                    router_grad=router_grad.detach().clone(),
                    expert_grads=expert_grads,
                )

            padded_result = run_case(
                layer=padded_layer,
                head=padded_head,
                hidden=padded_hidden,
                padding_mask=padded_padding_mask,
                packed_params=None,
                packed=False,
            )
            packed_result = run_case(
                layer=packed_layer,
                head=packed_head,
                hidden=packed_hidden,
                padding_mask=packed_padding_mask,
                packed_params=packed_seq_params,
                packed=True,
            )

            for result in (padded_result, packed_result):
                assert torch.equal(
                    result.padding_input_grad,
                    torch.zeros_like(result.padding_input_grad),
                )
            for padded_value, packed_value in (
                (padded_result.policy_loss, packed_result.policy_loss),
                (padded_result.auxiliary_loss, packed_result.auxiliary_loss),
                (padded_result.valid_logits, packed_result.valid_logits),
                (padded_result.valid_input_grad, packed_result.valid_input_grad),
                (padded_result.router_grad, packed_result.router_grad),
            ):
                torch.testing.assert_close(
                    padded_value,
                    packed_value,
                    atol=1e-5,
                    rtol=1e-5,
                )

            has_nonzero_expert_grad = False
            for name in expert_parameter_names:
                padded_grad = padded_result.expert_grads[name]
                packed_grad = packed_result.expert_grads[name]
                if padded_grad is None or packed_grad is None:
                    assert padded_grad is None and packed_grad is None
                    continue
                torch.testing.assert_close(
                    padded_grad,
                    packed_grad,
                    atol=1e-5,
                    rtol=1e-5,
                )
                has_nonzero_expert_grad |= bool(torch.count_nonzero(padded_grad).item())
            assert has_nonzero_expert_grad
        finally:
            destroy_moe_metrics_tracker()
            parallel_state.destroy_model_parallel()

    def test_sequence_packing_gradients(self):
        from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
        from nemo_rl.models.megatron.data import (
            _pack_sequences_for_megatron,
            make_processed_microbatch_iterator,
        )
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            forward_with_post_processing_fn,
        )

        # Initialize process group
        torch.distributed.init_process_group(backend="nccl")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Create CP group - all ranks participate in CP
        cp_group = torch.distributed.new_group(ranks=list(range(world_size)))

        # Patch get_context_parallel_group to always return cp_group
        # (Assume it's imported from nemo_rl.models.megatron.common)
        import megatron.core.parallel_state as parallel_state

        parallel_state._CONTEXT_PARALLEL_GROUP = cp_group
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = torch.distributed.new_group(
            ranks=[rank]
        )

        # Test parameters
        batch_size = 4
        max_seq_len = 512
        vocab_size = 1000
        cp_size = self.cp_size

        # Ensure sequence length is compatible with CP load balancing
        if max_seq_len % (2 * cp_size) != 0:
            max_seq_len = (max_seq_len // (2 * cp_size) + 1) * (2 * cp_size)

        # Create test data with varying sequence lengths
        torch.manual_seed(42)  # For reproducibility
        seq_lengths = torch.tensor(
            [
                max_seq_len // 4,
                max_seq_len * 1 // 4,
                max_seq_len // 4,
                max_seq_len * 3 // 4,
            ]
        )

        # Create input data
        input_ids = torch.zeros(
            batch_size, max_seq_len, dtype=torch.long, device="cuda"
        )
        token_mask = torch.zeros(
            batch_size, max_seq_len, dtype=torch.float, device="cuda"
        )

        # Fill with random tokens up to seq_length
        for i in range(batch_size):
            length = seq_lengths[i]
            input_ids[i, :length] = torch.randint(
                0, vocab_size, (length,), device="cuda"
            )
            token_mask[i, :length] = 1.0

        # Create other required tensors
        sample_mask = torch.ones(batch_size, dtype=torch.float, device="cuda")
        advantages = torch.randn(batch_size, max_seq_len, device="cuda")
        prev_logprobs = torch.randn(batch_size, max_seq_len, device="cuda")
        generation_logprobs = torch.randn(batch_size, max_seq_len, device="cuda")
        reference_policy_logprobs = generation_logprobs.clone()

        original_data = {
            "input_ids": input_ids,
            "input_lengths": seq_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": advantages,
            "prev_logprobs": prev_logprobs,
            "generation_logprobs": generation_logprobs,
            "reference_policy_logprobs": reference_policy_logprobs,
        }

        # ===== TEST 1: Baseline (no sequence packing) =====
        print(f"Rank {rank}: Testing baseline (no sequence packing)")

        baseline_logits = torch.randn(
            batch_size, max_seq_len, vocab_size, requires_grad=True, device="cuda"
        )

        loss_config = ClippedPGLossConfig(
            reference_policy_kl_penalty=0.1,
            ratio_clip_c=3.0,
        )

        base_loss_fn = ClippedPGLossFn(loss_config)
        data_dict = BatchedDataDict(original_data)

        global_valid_toks = torch.tensor(
            sum(seq_lengths).item(), dtype=torch.float, device="cuda"
        )
        global_valid_seqs = torch.tensor(batch_size, dtype=torch.float, device="cuda")

        # Forward pass
        loss_input, data_dict = prepare_loss_input(
            baseline_logits, data_dict, base_loss_fn
        )
        baseline_loss, _ = base_loss_fn(
            data=data_dict,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            **loss_input,
        )

        # Backward pass
        baseline_loss.backward()

        # Check baseline gradients
        baseline_grad_norm = torch.norm(baseline_logits.grad).item()
        baseline_grad_max = torch.max(torch.abs(baseline_logits.grad)).item()
        baseline_grad_mean = torch.mean(torch.abs(baseline_logits.grad)).item()
        baseline_grad_store = baseline_logits.grad.clone()
        baseline_logits.grad.zero_()

        print(
            f"Rank {rank}: Baseline gradient stats - norm: {baseline_grad_norm:.4f}, max: {baseline_grad_max:.4f}, mean: {baseline_grad_mean:.4f}"
        )

        # ===== TEST 2: Sequence packing with context parallelism =====
        print(f"Rank {rank}: Testing with sequence packing + CP")

        # Pack sequences
        pad_to_multiple = cp_size * 2  # Common requirement for CP
        (
            packed_input_ids,
            packed_input_ids_cp,
            packed_seq_params,
            cu_seqlens,
            cu_seqlens_padded,
        ) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths,
            pad_individual_seqs_to_multiple_of=pad_to_multiple,
            pad_packed_seq_to=max_seq_len * batch_size if cp_size > 1 else None,
            cp_rank=rank,
            cp_size=cp_size,
        )

        # For CP, logits are sharded across context parallel ranks
        def make_packed_logits(logits):
            packed_logits = torch.zeros(
                1, packed_input_ids_cp.shape[1], vocab_size, device="cuda"
            )
            run_seq = 0
            for i, seq_len in enumerate(seq_lengths):
                padded_seqlen = cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]
                if padded_seqlen > baseline_logits.shape[1]:
                    # pad the logits with zeros
                    tmp_logits = torch.zeros(
                        1, padded_seqlen, vocab_size, device="cuda"
                    )
                    tmp_logits[:, :seq_len] = baseline_logits[i : i + 1, :seq_len]
                else:
                    tmp_logits = baseline_logits[i : i + 1, :padded_seqlen]
                packed_logits[
                    :, run_seq // cp_size : (run_seq + padded_seqlen) // cp_size, :
                ] = _get_tokens_on_this_cp_rank(tmp_logits, rank, cp_size)
                run_seq += padded_seqlen
            return packed_logits

        packed_logits = make_packed_logits(baseline_logits)

        # Create sequence packing wrapper
        tp_group = torch.distributed.new_group(ranks=[rank])
        wrapper = SequencePackingLossWrapper(
            loss_fn=base_loss_fn,
            prepare_fn=prepare_loss_input,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            vocab_parallel_rank=0,
            vocab_parallel_group=tp_group,
            context_parallel_group=cp_group,
        )

        # Create data dict for packed sequences
        packed_data_dict = BatchedDataDict(original_data)

        # Forward pass
        packed_loss, _ = wrapper(
            packed_logits,
            packed_data_dict,
            global_valid_seqs,
            global_valid_toks,
        )

        # Backward pass
        packed_loss /= cp_size
        packed_loss.backward()

        # Check packed gradients
        packed_grad = baseline_logits.grad.clone()
        # all-reduce across cp ranks
        torch.distributed.all_reduce(packed_grad, op=torch.distributed.ReduceOp.SUM)

        packed_grad_norm = torch.norm(packed_grad).item()
        packed_grad_max = torch.max(torch.abs(packed_grad)).item()
        packed_grad_mean = torch.mean(torch.abs(packed_grad)).item()

        print(
            f"Rank {rank}: Packed gradient stats - norm: {packed_grad_norm:.4f}, max: {packed_grad_max:.4f}, mean: {packed_grad_mean:.4f}"
        )

        # ===== ANALYSIS =====
        gradient_ratio_norm = (
            packed_grad_norm / baseline_grad_norm
            if baseline_grad_norm > 0
            else float("inf")
        )
        gradient_ratio_max = (
            packed_grad_max / baseline_grad_max
            if baseline_grad_max > 0
            else float("inf")
        )
        gradient_ratio_mean = (
            packed_grad_mean / baseline_grad_mean
            if baseline_grad_mean > 0
            else float("inf")
        )

        print(
            f"Rank {rank}: Gradient ratios - norm: {gradient_ratio_norm:.4f}, max: {gradient_ratio_max:.4f}, mean: {gradient_ratio_mean:.4f}"
        )

        print(
            f"differences by token: {torch.sum(torch.abs(packed_grad - baseline_grad_store), dim=-1)}"
        )

        torch.testing.assert_close(
            packed_grad, baseline_grad_store, atol=1e-5, rtol=1e-5
        )

        # test 3: with forward_with_post_processing_fn
        # reset grad
        baseline_logits.grad.zero_()
        packed_logits = make_packed_logits(baseline_logits)

        # mock straggler detector with dummy context manager
        mock_straggler_timer = MagicMock()
        mock_straggler_timer.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )

        # mock model forward
        class MockModel:
            def __init__(self):
                self.logits = packed_logits

            def __call__(self, *args, **kwargs):
                return self.logits

            def forward(
                self, input_ids, position_ids, attention_mask, packed_seq_params=None
            ):
                return self.logits

        cfg = {
            "sequence_packing": {"enabled": True},
            "dynamic_batching": {"enabled": False},
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": cp_size,
            },
        }

        post_processor = LossPostProcessor(
            loss_fn=base_loss_fn,
            cfg=cfg,
            cp_normalize=True,
        )

        output_tensor, wrapped_loss_fn = forward_with_post_processing_fn(
            data_iterator=make_processed_microbatch_iterator(
                iter([packed_data_dict]),
                cfg=cfg,
                seq_length_key="input_lengths",
                pad_individual_seqs_to_multiple_of=pad_to_multiple,
                pad_packed_seq_to_multiple_of=1,
                straggler_timer=mock_straggler_timer,
                pad_full_seq_to=max_seq_len * batch_size if cp_size > 1 else None,
            ),
            model=MockModel(),
            post_processing_fn=post_processor,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            straggler_timer=mock_straggler_timer,
        )
        loss, _ = wrapped_loss_fn(output_tensor)

        loss.backward()

        # Check packed gradients
        packed_grad = baseline_logits.grad.clone()
        # all-reduce across cp ranks
        torch.distributed.all_reduce(packed_grad, op=torch.distributed.ReduceOp.SUM)

        packed_grad_norm = torch.norm(packed_grad).item()
        packed_grad_max = torch.max(torch.abs(packed_grad)).item()
        packed_grad_mean = torch.mean(torch.abs(packed_grad)).item()
        print(
            f"Rank {rank}: Packed gradient stats - norm: {packed_grad_norm:.4f}, max: {packed_grad_max:.4f}, mean: {packed_grad_mean:.4f}"
        )

        gradient_ratio_norm = (
            packed_grad_norm / baseline_grad_norm
            if baseline_grad_norm > 0
            else float("inf")
        )
        gradient_ratio_max = (
            packed_grad_max / baseline_grad_max
            if baseline_grad_max > 0
            else float("inf")
        )

        print(
            f"Rank {rank}: Gradient ratios - norm: {gradient_ratio_norm:.4f}, max: {gradient_ratio_max:.4f}"
        )
        print(
            f"differences by token: {torch.sum(torch.abs(packed_grad - baseline_grad_store), dim=-1)}"
        )

        if cp_size == 1:
            self._test_packed_policy_gradient_ownership()
